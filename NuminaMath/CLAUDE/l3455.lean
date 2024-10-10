import Mathlib

namespace recurrence_implies_general_formula_l3455_345520

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → (n * a n - 2 * a (n + 1)) / a (n + 1) = n

/-- The general formula for the sequence -/
def GeneralFormula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 / (n * (n + 1))

theorem recurrence_implies_general_formula (a : ℕ → ℝ) :
  RecurrenceSequence a → GeneralFormula a := by
  sorry

end recurrence_implies_general_formula_l3455_345520


namespace john_hired_twenty_lessons_l3455_345504

/-- Given the cost of a piano, the original price of a lesson, the discount percentage,
    and the total cost, calculate the number of lessons hired. -/
def number_of_lessons (piano_cost lesson_price discount_percent total_cost : ℚ) : ℚ :=
  let discounted_price := lesson_price * (1 - discount_percent / 100)
  let lesson_cost := total_cost - piano_cost
  lesson_cost / discounted_price

/-- Prove that given the specified costs and discount, John hired 20 lessons. -/
theorem john_hired_twenty_lessons :
  number_of_lessons 500 40 25 1100 = 20 := by
  sorry

end john_hired_twenty_lessons_l3455_345504


namespace problem_1_problem_2_l3455_345598

-- Problem 1
theorem problem_1 : (1/2)⁻¹ - 2 * Real.tan (45 * π / 180) + |1 - Real.sqrt 2| = Real.sqrt 2 - 1 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a = Real.sqrt 3 + 2) : 
  (a / (a^2 - 4) + 1 / (2 - a)) / ((2*a + 4) / (a^2 + 4*a + 4)) = -Real.sqrt 3 / 3 := by sorry

end problem_1_problem_2_l3455_345598


namespace pencils_per_box_l3455_345571

theorem pencils_per_box (num_boxes : ℕ) (pencils_given : ℕ) (pencils_left : ℕ) :
  num_boxes = 2 ∧ pencils_given = 15 ∧ pencils_left = 9 →
  (pencils_given + pencils_left) / num_boxes = 12 := by
  sorry

end pencils_per_box_l3455_345571


namespace six_balls_four_boxes_l3455_345556

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there are 9 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 9 := by
  sorry

end six_balls_four_boxes_l3455_345556


namespace divisor_problem_l3455_345508

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 199 →
  quotient = 11 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 18 := by
sorry

end divisor_problem_l3455_345508


namespace solve_equation_l3455_345595

theorem solve_equation (x : ℝ) : 2*x - 3*x + 4*x = 120 → x = 40 := by
  sorry

end solve_equation_l3455_345595


namespace isosceles_triangle_not_unique_l3455_345548

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base_angle : ℝ
  other_angle : ℝ

/-- A function that attempts to construct an isosceles triangle from given angles -/
noncomputable def construct_isosceles_triangle (ba oa : ℝ) : Option IsoscelesTriangle := sorry

/-- Theorem stating that an isosceles triangle is not uniquely determined by a base angle and another angle -/
theorem isosceles_triangle_not_unique :
  ∃ (ba₁ oa₁ ba₂ oa₂ : ℝ),
    ba₁ = ba₂ ∧
    oa₁ = oa₂ ∧
    construct_isosceles_triangle ba₁ oa₁ ≠ construct_isosceles_triangle ba₂ oa₂ :=
sorry

end isosceles_triangle_not_unique_l3455_345548


namespace statement_is_false_l3455_345555

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem statement_is_false : ∃ n : ℕ, 
  (sum_of_digits n % 6 = 0) ∧ (n % 6 ≠ 0) := by sorry

end statement_is_false_l3455_345555


namespace donation_proof_l3455_345510

/-- The amount donated to Animal Preservation Park -/
def animal_park_donation : ℝ := sorry

/-- The amount donated to Treetown National Park and The Forest Reserve combined -/
def combined_donation : ℝ := animal_park_donation + 140

/-- The total donation to all three parks -/
def total_donation : ℝ := 1000

theorem donation_proof : combined_donation = 570 := by
  sorry

end donation_proof_l3455_345510


namespace sequence_transformation_l3455_345599

/-- Represents a sequence of letters 'A' and 'B' -/
def Sequence := List Char

/-- An operation that can be performed on a sequence -/
inductive Operation
| Insert (c : Char) (pos : Nat) (count : Nat)
| Remove (pos : Nat) (count : Nat)

/-- Applies an operation to a sequence -/
def applyOperation (s : Sequence) (op : Operation) : Sequence :=
  match op with
  | Operation.Insert c pos count => sorry
  | Operation.Remove pos count => sorry

/-- Checks if a sequence contains only 'A' and 'B' -/
def isValidSequence (s : Sequence) : Prop :=
  s.all (fun c => c = 'A' ∨ c = 'B')

/-- Theorem: Any two valid sequences of length 100 can be transformed
    into each other using at most 100 operations -/
theorem sequence_transformation
  (s1 s2 : Sequence)
  (h1 : s1.length = 100)
  (h2 : s2.length = 100)
  (v1 : isValidSequence s1)
  (v2 : isValidSequence s2) :
  ∃ (ops : List Operation),
    ops.length ≤ 100 ∧
    (ops.foldl applyOperation s1 = s2) :=
  sorry

end sequence_transformation_l3455_345599


namespace sum_S_six_cards_l3455_345534

/-- The number of strictly increasing subsequences of length 2 or more in a sequence -/
def S (π : List ℕ) : ℕ := sorry

/-- The sum of S(π) over all permutations of n distinct elements -/
def sum_S (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of S(π) over all permutations of 6 distinct elements is 8287 -/
theorem sum_S_six_cards : sum_S 6 = 8287 := by sorry

end sum_S_six_cards_l3455_345534


namespace ellipse_major_axis_length_l3455_345567

/-- The ellipse defined by x^2/9 + y^2/4 = 1 -/
def ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) + (p.2^2 / 4) = 1}

/-- The length of the major axis of the ellipse -/
def major_axis_length : ℝ := 6

/-- Theorem: The length of the major axis of the ellipse defined by x^2/9 + y^2/4 = 1 is 6 -/
theorem ellipse_major_axis_length : 
  ∀ p ∈ ellipse, major_axis_length = 6 := by
  sorry

end ellipse_major_axis_length_l3455_345567


namespace number_puzzle_l3455_345535

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 15) = 75 := by
  sorry

end number_puzzle_l3455_345535


namespace triangle_cosine_identities_l3455_345533

theorem triangle_cosine_identities (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  (Real.cos (2 * α) + Real.cos (2 * β) + Real.cos (2 * γ) + 4 * Real.cos α * Real.cos β * Real.cos γ + 1 = 0) ∧
  (Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 + 2 * Real.cos α * Real.cos β * Real.cos γ = 1) := by
  sorry

end triangle_cosine_identities_l3455_345533


namespace log_sum_equality_l3455_345551

theorem log_sum_equality : Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) + 8^(2/3) = 6 := by
  sorry

end log_sum_equality_l3455_345551


namespace squared_sum_product_l3455_345569

theorem squared_sum_product (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 108) :
  a^2 * b + a * b^2 = 108 := by sorry

end squared_sum_product_l3455_345569


namespace village_population_after_events_l3455_345565

theorem village_population_after_events (initial_population : ℕ) : 
  initial_population = 7800 → 
  (initial_population - initial_population / 10 - 
   (initial_population - initial_population / 10) / 4) = 5265 := by
sorry

end village_population_after_events_l3455_345565


namespace intersection_equals_B_implies_a_is_one_l3455_345574

def A : Set ℝ := {-1, 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 1}

theorem intersection_equals_B_implies_a_is_one (a : ℝ) : A ∩ B a = B a → a = 1 := by
  sorry

end intersection_equals_B_implies_a_is_one_l3455_345574


namespace tens_digit_of_2020_pow_2021_minus_2022_l3455_345501

theorem tens_digit_of_2020_pow_2021_minus_2022 : ∃ n : ℕ, 
  (2020^2021 - 2022) % 100 = 70 + n ∧ n < 10 :=
by sorry

end tens_digit_of_2020_pow_2021_minus_2022_l3455_345501


namespace work_completion_time_l3455_345579

/-- The number of days it takes A to complete the work alone -/
def days_A : ℝ := 6

/-- The total payment for the work -/
def total_payment : ℝ := 4000

/-- The number of days it takes A, B, and C to complete the work together -/
def days_ABC : ℝ := 3

/-- The payment to C -/
def payment_C : ℝ := 500.0000000000002

/-- The number of days it takes B to complete the work alone -/
def days_B : ℝ := 8

theorem work_completion_time :
  (1 / days_A + 1 / days_B + payment_C / total_payment * (1 / days_ABC) = 1 / days_ABC) ∧
  days_B = 8 := by sorry

end work_completion_time_l3455_345579


namespace dice_surface_area_l3455_345523

/-- The surface area of a cube with edge length 20 centimeters is 2400 square centimeters. -/
theorem dice_surface_area :
  let edge_length : ℝ := 20
  let surface_area : ℝ := 6 * edge_length ^ 2
  surface_area = 2400 := by sorry

end dice_surface_area_l3455_345523


namespace weight_relationships_l3455_345515

/-- Given the weights of Brenda, Mel, and Tom, prove their relationships and specific weights. -/
theorem weight_relationships (B M T : ℕ) : 
  B = 3 * M + 10 →  -- Brenda weighs 10 pounds more than 3 times Mel's weight
  T = 2 * M →       -- Tom weighs twice as much as Mel
  2 * T = B →       -- Tom weighs half as much as Brenda
  B = 220 →         -- Brenda weighs 220 pounds
  M = 70 ∧ T = 140  -- Prove that Mel weighs 70 pounds and Tom weighs 140 pounds
:= by sorry

end weight_relationships_l3455_345515


namespace slope_angle_vertical_line_l3455_345578

/-- Given two points A(2, 1) and B(2, 3), prove that the slope angle of the line AB is 90 degrees. -/
theorem slope_angle_vertical_line : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (2, 3)
  let slope_angle := Real.arctan ((B.2 - A.2) / (B.1 - A.1)) * (180 / Real.pi)
  slope_angle = 90 := by
  sorry

end slope_angle_vertical_line_l3455_345578


namespace total_rope_length_l3455_345532

def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss : ℝ := 1.2

theorem total_rope_length :
  let initial_length := rope_lengths.sum
  let num_knots := rope_lengths.length - 1
  let total_loss := num_knots * knot_loss
  initial_length - total_loss = 35 := by
  sorry

end total_rope_length_l3455_345532


namespace resistance_change_l3455_345503

/-- Represents the change in resistance when a switch is closed in a circuit with three resistors. -/
theorem resistance_change (R₁ R₂ R₃ : ℝ) (h₁ : R₁ = 1) (h₂ : R₂ = 2) (h₃ : R₃ = 4) :
  ∃ (ε : ℝ), abs (R₁ + (R₂ * R₃) / (R₂ + R₃) - R₁ + 0.67) < ε ∧ ε > 0 := by
  sorry

end resistance_change_l3455_345503


namespace room_area_from_carpet_l3455_345505

/-- Given a rectangular carpet covering 30% of a room's floor area, 
    if the carpet measures 4 feet by 9 feet, 
    then the total floor area of the room is 120 square feet. -/
theorem room_area_from_carpet (carpet_length carpet_width : ℝ) 
  (carpet_coverage_percent : ℝ) (total_area : ℝ) :
  carpet_length = 4 →
  carpet_width = 9 →
  carpet_coverage_percent = 30 →
  carpet_length * carpet_width / total_area = carpet_coverage_percent / 100 →
  total_area = 120 :=
by sorry

end room_area_from_carpet_l3455_345505


namespace cakes_slices_problem_l3455_345563

theorem cakes_slices_problem (total_slices : ℕ) (friends_fraction : ℚ) 
  (family_fraction : ℚ) (eaten_slices : ℕ) (remaining_slices : ℕ) :
  total_slices = 16 →
  family_fraction = 1/3 →
  eaten_slices = 3 →
  remaining_slices = 5 →
  (1 - friends_fraction) * (1 - family_fraction) * total_slices - eaten_slices = remaining_slices →
  friends_fraction = 1/4 := by
sorry

end cakes_slices_problem_l3455_345563


namespace symmetric_line_wrt_point_symmetric_line_wrt_line_l3455_345572

-- Define the original line l
def l (x y : ℝ) : Prop := y = 2 * x + 1

-- Define the point M
def M : ℝ × ℝ := (3, 2)

-- Define the line to be reflected
def line_to_reflect (x y : ℝ) : Prop := x - y - 2 = 0

-- Statement for the first part of the problem
theorem symmetric_line_wrt_point :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    (∀ (x' y' : ℝ), l x' y' → 
      (x + x') / 2 = M.1 ∧ (y + y') / 2 = M.2) →
    y = a * x + b ↔ y = 2 * x - 9 :=
sorry

-- Statement for the second part of the problem
theorem symmetric_line_wrt_line :
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    (∀ (x' y' : ℝ), line_to_reflect x' y' → 
      ∃ (x'' y'' : ℝ), l ((x + x'') / 2) ((y + y'') / 2) ∧
      (y'' - y) / (x'' - x) = -1 / (2 : ℝ)) →
    a * x + b * y + c = 0 ↔ 7 * x - y + 16 = 0 :=
sorry

end symmetric_line_wrt_point_symmetric_line_wrt_line_l3455_345572


namespace painted_cube_theorem_l3455_345557

theorem painted_cube_theorem (n : ℕ) (h1 : n > 4) :
  (2 * (n - 2) = n^2 - 2*n + 1) → n = 5 := by
  sorry

end painted_cube_theorem_l3455_345557


namespace prime_even_intersection_l3455_345575

def isPrime (n : ℕ) : Prop := sorry

def isEven (n : ℕ) : Prop := sorry

def P : Set ℕ := {n | isPrime n}
def Q : Set ℕ := {n | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by sorry

end prime_even_intersection_l3455_345575


namespace manager_selection_l3455_345552

theorem manager_selection (n m k : ℕ) (h1 : n = 8) (h2 : m = 4) (h3 : k = 2) : 
  (n.choose m) - ((n - k).choose (m - k)) = 55 := by
  sorry

end manager_selection_l3455_345552


namespace union_when_a_is_two_intersection_empty_iff_l3455_345561

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 3 ∧ a > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- Theorem 1: When a = 2, A ∪ B = {x | -2 < x < 7}
theorem union_when_a_is_two : 
  A 2 ∪ B = {x : ℝ | -2 < x ∧ x < 7} := by sorry

-- Theorem 2: A ∩ B = ∅ if and only if a ≥ 5
theorem intersection_empty_iff : 
  ∀ a : ℝ, A a ∩ B = ∅ ↔ a ≥ 5 := by sorry

end union_when_a_is_two_intersection_empty_iff_l3455_345561


namespace maude_age_l3455_345573

theorem maude_age (anne emile maude : ℕ) 
  (h1 : anne = 96)
  (h2 : anne = 2 * emile)
  (h3 : emile = 6 * maude) :
  maude = 8 := by
sorry

end maude_age_l3455_345573


namespace jesse_book_reading_l3455_345594

theorem jesse_book_reading (pages_read pages_left : ℕ) 
  (h1 : pages_read = 83) 
  (h2 : pages_left = 166) : 
  (pages_read : ℚ) / (pages_read + pages_left) = 1 / 3 := by
  sorry

end jesse_book_reading_l3455_345594


namespace square_cut_perimeter_l3455_345554

theorem square_cut_perimeter (square_side : ℝ) (total_perimeter : ℝ) :
  square_side = 4 →
  total_perimeter = 25 →
  ∃ (rect1_length rect1_width rect2_length rect2_width : ℝ),
    rect1_length * rect1_width + rect2_length * rect2_width = square_side * square_side ∧
    2 * (rect1_length + rect1_width) + 2 * (rect2_length + rect2_width) = total_perimeter :=
by sorry

end square_cut_perimeter_l3455_345554


namespace angle_terminal_side_value_l3455_345518

theorem angle_terminal_side_value (m : ℝ) (α : ℝ) (h : m ≠ 0) :
  (∃ (x y : ℝ), x = -4 * m ∧ y = 3 * m ∧ 
    x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
    y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  2 * Real.sin α + Real.cos α = 2/5 ∨ 2 * Real.sin α + Real.cos α = -2/5 :=
by sorry

end angle_terminal_side_value_l3455_345518


namespace inequality_solution_l3455_345524

theorem inequality_solution (y : ℝ) : 
  (7/30 : ℝ) + |y - 3/10| < 11/30 ↔ 1/6 < y ∧ y < 1/3 :=
by sorry

end inequality_solution_l3455_345524


namespace intersection_A_complement_B_l3455_345586

def U : Set ℝ := Set.univ

def A : Set ℝ := {-3, -1, 0, 1, 3}

def B : Set ℝ := {x | |x - 1| > 1}

theorem intersection_A_complement_B : A ∩ (U \ B) = {0, 1} := by
  sorry

end intersection_A_complement_B_l3455_345586


namespace x_plus_y_value_l3455_345559

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : x + |y| - y = 12) : 
  x + y = 18/5 := by
  sorry

end x_plus_y_value_l3455_345559


namespace bd_length_is_six_l3455_345511

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem bd_length_is_six (ABCD : Quadrilateral) : 
  length ABCD.A ABCD.B = 6 →
  length ABCD.B ABCD.C = 11 →
  length ABCD.C ABCD.D = 6 →
  length ABCD.D ABCD.A = 8 →
  ∃ n : ℕ, length ABCD.B ABCD.D = n →
  length ABCD.B ABCD.D = 6 := by
  sorry

end bd_length_is_six_l3455_345511


namespace empty_solution_set_range_l3455_345597

theorem empty_solution_set_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 2 * m * x + 1 ≥ 0) ↔ m ∈ Set.Icc 0 1 := by
  sorry

end empty_solution_set_range_l3455_345597


namespace power_nine_2023_mod_50_l3455_345514

theorem power_nine_2023_mod_50 : 9^2023 % 50 = 29 := by
  sorry

end power_nine_2023_mod_50_l3455_345514


namespace max_player_salary_l3455_345560

theorem max_player_salary (n : ℕ) (min_salary max_total : ℝ) :
  n = 15 →
  min_salary = 20000 →
  max_total = 800000 →
  (∃ (salaries : Fin n → ℝ),
    (∀ i, salaries i ≥ min_salary) ∧
    (Finset.sum Finset.univ salaries ≤ max_total) ∧
    (∀ i, salaries i ≤ 520000)) ∧
  ¬(∃ (salaries : Fin n → ℝ),
    (∀ i, salaries i ≥ min_salary) ∧
    (Finset.sum Finset.univ salaries ≤ max_total) ∧
    (∃ i, salaries i > 520000)) :=
by sorry

end max_player_salary_l3455_345560


namespace bakers_sales_l3455_345585

/-- Baker's cake and pastry sales problem -/
theorem bakers_sales (cakes_made pastries_made cakes_sold pastries_sold : ℕ) 
  (h1 : cakes_made = 475)
  (h2 : pastries_made = 539)
  (h3 : cakes_sold = 358)
  (h4 : pastries_sold = 297) :
  cakes_sold - pastries_sold = 61 := by
  sorry

end bakers_sales_l3455_345585


namespace inconsistent_division_problem_l3455_345527

theorem inconsistent_division_problem 
  (x y q : ℕ+) 
  (h1 : x = 9 * y + 4)
  (h2 : 2 * x = 7 * q + 1)
  (h3 : 5 * y - x = 3) :
  False :=
sorry

end inconsistent_division_problem_l3455_345527


namespace coloring_books_shelves_l3455_345549

/-- Calculates the number of shelves needed to display remaining coloring books --/
def shelves_needed (initial_stock : ℕ) (sold : ℕ) (donated : ℕ) (books_per_shelf : ℕ) : ℕ :=
  ((initial_stock - sold - donated) + books_per_shelf - 1) / books_per_shelf

/-- Theorem stating that given the problem conditions, 6 shelves are needed --/
theorem coloring_books_shelves :
  shelves_needed 150 55 30 12 = 6 := by
  sorry

end coloring_books_shelves_l3455_345549


namespace muffin_price_theorem_l3455_345537

/-- Promotional sale: Buy three muffins at regular price, get fourth muffin free -/
def promotional_sale (regular_price : ℝ) : ℝ := 3 * regular_price

/-- The total amount John paid for four muffins -/
def total_paid : ℝ := 15

theorem muffin_price_theorem :
  ∃ (regular_price : ℝ), promotional_sale regular_price = total_paid ∧ regular_price = 5 := by
  sorry

end muffin_price_theorem_l3455_345537


namespace canteen_distance_l3455_345562

theorem canteen_distance (a b c : ℝ) (h1 : a = 360) (h2 : c = 800) (h3 : a^2 + b^2 = c^2) :
  b / 2 = 438.6 := by sorry

end canteen_distance_l3455_345562


namespace horner_v3_value_l3455_345544

/-- Horner's Method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x⁶ - 5x⁵ + 6x⁴ + x² + 0.3x + 2 -/
def f (x : ℝ) : ℝ :=
  x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, -5, 6, 0, 1, 0.3, 2]

/-- Theorem: v₃ = -40 when evaluating f(-2) using Horner's Method -/
theorem horner_v3_value :
  let x := -2
  let v₀ := 1
  let v₁ := v₀ * x + f_coeffs[1]!
  let v₂ := v₁ * x + f_coeffs[2]!
  let v₃ := v₂ * x + f_coeffs[3]!
  v₃ = -40 := by sorry

end horner_v3_value_l3455_345544


namespace five_regular_polyhedra_l3455_345587

/-- A convex regular polyhedron with n edges meeting at each vertex and k sides on each face. -/
structure ConvexRegularPolyhedron where
  n : ℕ
  k : ℕ
  n_ge_three : n ≥ 3
  k_ge_three : k ≥ 3

/-- The inequality that must be satisfied by a convex regular polyhedron. -/
def satisfies_inequality (p : ConvexRegularPolyhedron) : Prop :=
  (1 : ℚ) / p.n + (1 : ℚ) / p.k > (1 : ℚ) / 2

/-- The theorem stating that there are only five types of convex regular polyhedra. -/
theorem five_regular_polyhedra :
  ∀ p : ConvexRegularPolyhedron, satisfies_inequality p ↔
    (p.n = 3 ∧ p.k = 3) ∨
    (p.n = 3 ∧ p.k = 4) ∨
    (p.n = 3 ∧ p.k = 5) ∨
    (p.n = 4 ∧ p.k = 3) ∨
    (p.n = 5 ∧ p.k = 3) :=
by sorry

end five_regular_polyhedra_l3455_345587


namespace fraction_equality_l3455_345500

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 8 * y) = 3) : 
  (2 * x + 8 * y) / (8 * x - 2 * y) = 1 / 3 := by
  sorry

end fraction_equality_l3455_345500


namespace computer_contract_probability_l3455_345522

theorem computer_contract_probability (p_hardware : ℚ) (p_not_software : ℚ) (p_at_least_one : ℚ)
  (h1 : p_hardware = 3 / 4)
  (h2 : p_not_software = 3 / 5)
  (h3 : p_at_least_one = 5 / 6) :
  p_hardware + (1 - p_not_software) - p_at_least_one = 19 / 60 :=
by sorry

end computer_contract_probability_l3455_345522


namespace triangle_area_from_smaller_triangles_l3455_345588

/-- Given a triangle divided into six parts by lines parallel to its sides,
    this theorem states that the area of the original triangle is equal to
    (√t₁ + √t₂ + √t₃)², where t₁, t₂, and t₃ are the areas of three of the
    smaller triangles formed. -/
theorem triangle_area_from_smaller_triangles 
  (t₁ t₂ t₃ : ℝ) 
  (h₁ : t₁ > 0) 
  (h₂ : t₂ > 0) 
  (h₃ : t₃ > 0) :
  ∃ T : ℝ, T > 0 ∧ T = (Real.sqrt t₁ + Real.sqrt t₂ + Real.sqrt t₃)^2 := by
  sorry

end triangle_area_from_smaller_triangles_l3455_345588


namespace area_of_triangle_PMF_l3455_345507

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  directrix : ℝ
  focus : ℝ × ℝ

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- The foot of the perpendicular from a point to the directrix -/
def footOfPerpendicular (p : Parabola) (point : PointOnParabola p) : ℝ × ℝ :=
  (p.directrix, point.y)

/-- The theorem stating the area of the triangle PMF -/
theorem area_of_triangle_PMF (p : Parabola) (P : PointOnParabola p) :
  p.equation = (fun x y => y^2 = 4*x) →
  p.directrix = -1 →
  p.focus = (1, 0) →
  (P.x - p.directrix)^2 + P.y^2 = 5^2 →
  let M := footOfPerpendicular p P
  let F := p.focus
  let area := (1/2) * |P.y| * 5
  area = 10 := by
  sorry

end area_of_triangle_PMF_l3455_345507


namespace stock_price_uniqueness_l3455_345542

theorem stock_price_uniqueness (n : Nat) (k l : Nat) (h_n : 0 < n ∧ n < 100) :
  (1 + n / 100 : ℚ) ^ k * (1 - n / 100 : ℚ) ^ l ≠ 1 := by
  sorry

end stock_price_uniqueness_l3455_345542


namespace soccer_ball_inflation_l3455_345512

/-- Proves that Ermias inflated 5 more balls than Alexia given the problem conditions -/
theorem soccer_ball_inflation (inflation_time ball_count_alexia total_time : ℕ) 
  (h1 : inflation_time = 20)
  (h2 : ball_count_alexia = 20)
  (h3 : total_time = 900) : 
  ∃ (additional_balls : ℕ), 
    inflation_time * ball_count_alexia + 
    inflation_time * (ball_count_alexia + additional_balls) = total_time ∧ 
    additional_balls = 5 := by
  sorry

end soccer_ball_inflation_l3455_345512


namespace total_average_donation_l3455_345592

/-- Represents the donation statistics for two units A and B -/
structure DonationStats where
  avg_donation_A : ℝ
  num_people_A : ℕ
  num_people_B : ℕ

/-- The conditions of the donation problem -/
def donation_conditions (stats : DonationStats) : Prop :=
  -- Unit B donated twice as much as unit A
  (stats.avg_donation_A * stats.num_people_A) * 2 = (stats.avg_donation_A - 100) * stats.num_people_B
  -- The average donation per person in unit B is $100 less than the average donation per person in unit A
  ∧ (stats.avg_donation_A - 100) > 0
  -- The number of people in unit A is one-fourth of the number of people in unit B
  ∧ stats.num_people_A * 4 = stats.num_people_B

/-- The theorem stating that the total average donation is $120 -/
theorem total_average_donation (stats : DonationStats) 
  (h : donation_conditions stats) : 
  (stats.avg_donation_A * stats.num_people_A + (stats.avg_donation_A - 100) * stats.num_people_B) / 
  (stats.num_people_A + stats.num_people_B) = 120 := by
  sorry


end total_average_donation_l3455_345592


namespace screw_nut_production_l3455_345590

theorem screw_nut_production (total_workers : ℕ) (screws_per_worker : ℕ) (nuts_per_worker : ℕ) 
  (screw_workers : ℕ) (nut_workers : ℕ) : 
  total_workers = 22 →
  screws_per_worker = 1200 →
  nuts_per_worker = 2000 →
  screw_workers = 10 →
  nut_workers = 12 →
  screw_workers + nut_workers = total_workers ∧
  2 * (screw_workers * screws_per_worker) = nut_workers * nuts_per_worker :=
by
  sorry

#check screw_nut_production

end screw_nut_production_l3455_345590


namespace x_value_l3455_345525

theorem x_value (x y : ℝ) 
  (h1 : x - y = 8)
  (h2 : x + y = 16)
  (h3 : x * y = 48) : x = 12 := by
  sorry

end x_value_l3455_345525


namespace jakes_weight_l3455_345540

theorem jakes_weight (jake_weight sister_weight : ℝ) : 
  (0.8 * jake_weight = 2 * sister_weight) →
  (jake_weight + sister_weight = 168) →
  (jake_weight = 120) := by
sorry

end jakes_weight_l3455_345540


namespace tangent_line_is_correct_l3455_345509

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Define the point of tangency
def point : ℝ × ℝ := (1, 2)

-- Define the proposed tangent line
def tangent_line (x y : ℝ) : Prop := 4*x - y - 2 = 0

-- Theorem statement
theorem tangent_line_is_correct :
  let (x₀, y₀) := point
  (∀ x, tangent_line x (f x)) ∧
  (tangent_line x₀ y₀) ∧
  (∀ x, x ≠ x₀ → ¬(tangent_line x (f x) ∧ tangent_line x₀ y₀)) :=
sorry

end tangent_line_is_correct_l3455_345509


namespace water_jar_problem_l3455_345566

theorem water_jar_problem (c_s c_l : ℝ) (h1 : c_s > 0) (h2 : c_l > 0) (h3 : c_s ≠ c_l) : 
  (1 / 6 : ℝ) * c_s = (1 / 5 : ℝ) * c_l → 
  (1 / 5 : ℝ) + (1 / 6 : ℝ) * c_s / c_l = (2 / 5 : ℝ) := by
  sorry

#check water_jar_problem

end water_jar_problem_l3455_345566


namespace library_theorem_l3455_345519

def library_problem (total_books : ℕ) (books_per_student : ℕ) 
  (day1_students : ℕ) (day2_students : ℕ) (day3_students : ℕ) : ℕ :=
  let books_remaining := total_books - 
    (day1_students + day2_students + day3_students) * books_per_student
  books_remaining / books_per_student

theorem library_theorem : 
  library_problem 120 5 4 5 6 = 9 := by
  sorry

end library_theorem_l3455_345519


namespace largest_equal_cost_number_l3455_345506

/-- Calculates the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates the number of digits of a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Converts a positive integer to its binary representation -/
def to_binary (n : ℕ) : List ℕ := sorry

/-- Calculates the cost of transmitting a number using Option 1 -/
def cost_option1 (n : ℕ) : ℕ :=
  sum_of_digits n + num_digits n

/-- Calculates the cost of transmitting a number using Option 2 -/
def cost_option2 (n : ℕ) : ℕ :=
  let binary := to_binary n
  (binary.filter (· = 1)).length + (binary.filter (· = 0)).length + binary.length

/-- Checks if the costs are equal for both options -/
def costs_equal (n : ℕ) : Prop :=
  cost_option1 n = cost_option2 n

theorem largest_equal_cost_number :
  ∀ n : ℕ, n < 2000 → n > 1539 → ¬(costs_equal n) := by sorry

end largest_equal_cost_number_l3455_345506


namespace gift_contribution_ratio_l3455_345553

theorem gift_contribution_ratio : 
  let lisa_savings : ℚ := 1200
  let mother_contribution : ℚ := 3/5 * lisa_savings
  let total_needed : ℚ := 3760
  let shortfall : ℚ := 400
  let total_contributions : ℚ := total_needed - shortfall
  let brother_contribution : ℚ := total_contributions - lisa_savings - mother_contribution
  brother_contribution / mother_contribution = 2 := by sorry

end gift_contribution_ratio_l3455_345553


namespace sin_cos_identity_l3455_345531

theorem sin_cos_identity : 
  Real.sin (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (138 * π / 180) * Real.cos (72 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_identity_l3455_345531


namespace pyramid_volume_with_conditions_l3455_345517

/-- The volume of a right pyramid with a hexagonal base -/
noncomputable def pyramidVolume (totalSurfaceArea : ℝ) (triangularFaceRatio : ℝ) : ℝ :=
  let hexagonalBaseArea := totalSurfaceArea / 3
  let sideLength := Real.sqrt (320 / (3 * Real.sqrt 3))
  let triangularHeight := 160 / sideLength
  let pyramidHeight := Real.sqrt (triangularHeight^2 - (sideLength / 2)^2)
  (1 / 3) * hexagonalBaseArea * pyramidHeight

/-- Theorem: The volume of the pyramid with given conditions -/
theorem pyramid_volume_with_conditions :
  ∃ (V : ℝ), pyramidVolume 720 (1/3) = V :=
sorry

end pyramid_volume_with_conditions_l3455_345517


namespace magnitude_of_vector_sum_l3455_345536

/-- Given plane vectors a and b satisfying certain conditions, prove that the magnitude of their sum is √21. -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) : 
  ‖a‖ = 2 → 
  ‖b‖ = 3 → 
  a - b = (Real.sqrt 2, Real.sqrt 3) →
  ‖a + b‖ = Real.sqrt 21 := by
  sorry

end magnitude_of_vector_sum_l3455_345536


namespace parallel_planes_imply_parallel_lines_l3455_345558

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_imply_parallel_lines 
  (α β γ : Plane) (m n : Line) :
  α ≠ β → α ≠ γ → β ≠ γ →  -- Three different planes
  intersect α γ = m →      -- α ∩ γ = m
  intersect β γ = n →      -- β ∩ γ = n
  parallel_planes α β →    -- If α ∥ β
  parallel_lines m n :=    -- Then m ∥ n
by sorry

end parallel_planes_imply_parallel_lines_l3455_345558


namespace power_sum_equals_40_l3455_345596

theorem power_sum_equals_40 : (-2)^4 + (-2)^3 + (-2)^2 + (-2)^1 + 2^1 + 2^2 + 2^3 + 2^4 = 40 := by
  sorry

end power_sum_equals_40_l3455_345596


namespace adam_students_in_ten_years_l3455_345530

theorem adam_students_in_ten_years : 
  let students_per_year : ℕ := 50
  let first_year_students : ℕ := 40
  let total_years : ℕ := 10
  (total_years - 1) * students_per_year + first_year_students = 490 := by
  sorry

end adam_students_in_ten_years_l3455_345530


namespace b_work_fraction_proof_l3455_345583

/-- The fraction of a day that b works --/
def b_work_fraction : ℚ := 1 / 5

/-- The time it takes a and b together to complete the work (in days) --/
def together_time : ℚ := 12

/-- The time it takes a alone to complete the work (in days) --/
def a_alone_time : ℚ := 20

/-- The time it takes a and b together to complete the work when b works a fraction of a day (in days) --/
def partial_together_time : ℚ := 15

theorem b_work_fraction_proof :
  (1 / a_alone_time + b_work_fraction * (1 / together_time) = 1 / partial_together_time) ∧
  (b_work_fraction > 0) ∧ (b_work_fraction < 1) := by
  sorry

end b_work_fraction_proof_l3455_345583


namespace chocolate_bar_ratio_l3455_345516

theorem chocolate_bar_ratio (total pieces : ℕ) (michael paige mandy : ℕ) : 
  total = 60 →
  paige = (total - michael) / 2 →
  mandy = 15 →
  total = michael + paige + mandy →
  michael / total = 0 :=
by
  sorry

end chocolate_bar_ratio_l3455_345516


namespace factorization_theorem_l3455_345550

/-- The polynomial to be factored -/
def p (x : ℝ) : ℝ := x^2 + 6*x + 9 - 64*x^4

/-- The first factor of the factorization -/
def f1 (x : ℝ) : ℝ := -8*x^2 + x + 3

/-- The second factor of the factorization -/
def f2 (x : ℝ) : ℝ := 8*x^2 + x + 3

/-- Theorem stating that p(x) is equal to the product of f1(x) and f2(x) for all real x -/
theorem factorization_theorem : ∀ x : ℝ, p x = f1 x * f2 x := by
  sorry

end factorization_theorem_l3455_345550


namespace banana_bread_flour_calculation_l3455_345593

/-- Given the recipe for banana bread, calculate the number of cups of flour needed. -/
theorem banana_bread_flour_calculation 
  (flour_per_mush : ℚ)  -- Cups of flour per cup of mush
  (bananas_per_mush : ℚ)  -- Number of bananas per cup of mush
  (total_bananas : ℚ)  -- Total number of bananas used
  (h1 : flour_per_mush = 3)  -- 3 cups of flour per cup of mush
  (h2 : bananas_per_mush = 4)  -- 4 bananas make one cup of mush
  (h3 : total_bananas = 20)  -- Hannah uses 20 bananas
  : (total_bananas / bananas_per_mush) * flour_per_mush = 15 := by
  sorry

#check banana_bread_flour_calculation

end banana_bread_flour_calculation_l3455_345593


namespace wedding_drinks_l3455_345576

theorem wedding_drinks (total_guests : ℕ) (num_drink_types : ℕ) 
  (champagne_glasses_per_guest : ℕ) (champagne_servings_per_bottle : ℕ)
  (wine_glasses_per_guest : ℕ) (wine_servings_per_bottle : ℕ)
  (juice_glasses_per_guest : ℕ) (juice_servings_per_bottle : ℕ)
  (h1 : total_guests = 120)
  (h2 : num_drink_types = 3)
  (h3 : champagne_glasses_per_guest = 2)
  (h4 : champagne_servings_per_bottle = 6)
  (h5 : wine_glasses_per_guest = 1)
  (h6 : wine_servings_per_bottle = 5)
  (h7 : juice_glasses_per_guest = 1)
  (h8 : juice_servings_per_bottle = 4) :
  let guests_per_drink_type := total_guests / num_drink_types
  let juice_bottles_needed := (guests_per_drink_type * juice_glasses_per_guest + juice_servings_per_bottle - 1) / juice_servings_per_bottle
  juice_bottles_needed = 10 := by
sorry

end wedding_drinks_l3455_345576


namespace sum_of_first_15_natural_numbers_mod_11_l3455_345582

theorem sum_of_first_15_natural_numbers_mod_11 :
  (List.range 16).sum % 11 = 10 := by
  sorry

end sum_of_first_15_natural_numbers_mod_11_l3455_345582


namespace quadratic_complex_roots_l3455_345546

theorem quadratic_complex_roots : ∃ (z₁ z₂ : ℂ),
  z₁ = (3 + Real.sqrt 14) / 2 + Complex.I * Real.sqrt 14 / 7 ∧
  z₂ = (3 - Real.sqrt 14) / 2 - Complex.I * Real.sqrt 14 / 7 ∧
  z₁^2 - 3*z₁ + 2 = 3 - 2*Complex.I ∧
  z₂^2 - 3*z₂ + 2 = 3 - 2*Complex.I :=
by sorry

end quadratic_complex_roots_l3455_345546


namespace rectangle_area_below_line_l3455_345547

/-- Given a rectangle bounded by y = 2a, y = -b, x = -2c, and x = d, 
    where a, b, c, and d are positive real numbers, and a line y = x + a 
    intersecting the rectangle, this theorem states the area of the 
    rectangle below the line. -/
theorem rectangle_area_below_line 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let rectangle_area := (2*a + b) * (d + 2*c)
  let triangle_area := (1/2) * (d + 2*c + b + a) * (a + b + 2*c)
  rectangle_area - triangle_area = 
    (2*a + b) * (d + 2*c) - (1/2) * (d + 2*c + b + a) * (a + b + 2*c) := by
  sorry

end rectangle_area_below_line_l3455_345547


namespace grocery_store_inventory_l3455_345564

theorem grocery_store_inventory (ordered : ℕ) (sold : ℕ) (storeroom : ℕ) 
  (h1 : ordered = 4458)
  (h2 : sold = 1561)
  (h3 : storeroom = 575) :
  ordered - sold + storeroom = 3472 :=
by sorry

end grocery_store_inventory_l3455_345564


namespace symmetric_point_theorem_l3455_345568

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

/-- Definition of point symmetry with respect to a line --/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), line_of_symmetry x₀ y₀ ∧
    (x₀ - x₁ = x₂ - x₀) ∧ (y₀ - y₁ = y₂ - y₀)

/-- Theorem: The point (2, -2) is symmetric to (-1, 1) with respect to the line x-y-1=0 --/
theorem symmetric_point_theorem : symmetric_points (-1) 1 2 (-2) := by
  sorry

end symmetric_point_theorem_l3455_345568


namespace class_size_class_size_is_60_l3455_345577

theorem class_size (cafeteria_students : ℕ) (no_lunch_students : ℕ) : ℕ :=
  let bring_lunch_students := 3 * cafeteria_students
  let total_lunch_students := cafeteria_students + bring_lunch_students
  let total_students := total_lunch_students + no_lunch_students
  total_students

theorem class_size_is_60 : 
  class_size 10 20 = 60 := by sorry

end class_size_class_size_is_60_l3455_345577


namespace product_of_solutions_l3455_345541

theorem product_of_solutions : ∃ (x y : ℝ), 
  (abs x = 3 * (abs x - 2)) ∧ 
  (abs y = 3 * (abs y - 2)) ∧ 
  (x ≠ y) ∧ 
  (x * y = -9) :=
sorry

end product_of_solutions_l3455_345541


namespace investment_schemes_count_l3455_345584

/-- The number of projects to be invested -/
def num_projects : ℕ := 4

/-- The number of candidate cities -/
def num_cities : ℕ := 4

/-- The maximum number of projects that can be invested in a single city -/
def max_projects_per_city : ℕ := 2

/-- A function that calculates the number of ways to distribute projects among cities -/
def investment_schemes (projects : ℕ) (cities : ℕ) (max_per_city : ℕ) : ℕ := sorry

/-- Theorem stating that the number of investment schemes is 240 -/
theorem investment_schemes_count : 
  investment_schemes num_projects num_cities max_projects_per_city = 240 := by sorry

end investment_schemes_count_l3455_345584


namespace kim_coffee_time_l3455_345543

/-- Represents the time Kim spends on her morning routine -/
structure MorningRoutine where
  coffee_time : ℕ
  status_update_time_per_employee : ℕ
  payroll_update_time_per_employee : ℕ
  number_of_employees : ℕ
  total_time : ℕ

/-- Theorem stating that Kim spends 5 minutes making coffee -/
theorem kim_coffee_time (routine : MorningRoutine)
  (h1 : routine.status_update_time_per_employee = 2)
  (h2 : routine.payroll_update_time_per_employee = 3)
  (h3 : routine.number_of_employees = 9)
  (h4 : routine.total_time = 50)
  (h5 : routine.total_time = routine.coffee_time +
    routine.number_of_employees * (routine.status_update_time_per_employee +
    routine.payroll_update_time_per_employee)) :
  routine.coffee_time = 5 := by
  sorry

end kim_coffee_time_l3455_345543


namespace complement_intersection_eq_set_l3455_345528

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_eq_set : (Aᶜ ∩ Bᶜ) = {1, 4, 5} := by sorry

end complement_intersection_eq_set_l3455_345528


namespace complex_fraction_calculation_l3455_345502

theorem complex_fraction_calculation : 
  let expr1 := (5 / 8 * 3 / 7 + 1 / 4 * 2 / 6) - (2 / 3 * 1 / 4 - 1 / 5 * 4 / 9)
  let expr2 := 7 / 9 * 2 / 5 * 1 / 2 * 5040 + 1 / 3 * 3 / 8 * 9 / 11 * 4230
  (expr1 * expr2 : ℚ) = 336 := by
  sorry

end complex_fraction_calculation_l3455_345502


namespace find_number_l3455_345513

theorem find_number : ∃ x : ℤ, 27 * (x + 143) = 9693 ∧ x = 216 := by sorry

end find_number_l3455_345513


namespace closest_integer_to_cube_root_1728_l3455_345580

theorem closest_integer_to_cube_root_1728 : 
  ∀ n : ℤ, |n - (1728 : ℝ)^(1/3)| ≥ |12 - (1728 : ℝ)^(1/3)| :=
by
  sorry

end closest_integer_to_cube_root_1728_l3455_345580


namespace inevitable_not_random_l3455_345591

-- Define the Event type
inductive Event
| Random : Event
| Inevitable : Event
| Impossible : Event

-- Define properties of events
def mayOccur (e : Event) : Prop :=
  match e with
  | Event.Random => true
  | Event.Inevitable => true
  | Event.Impossible => false

def willDefinitelyOccur (e : Event) : Prop :=
  match e with
  | Event.Inevitable => true
  | _ => false

-- Theorem: An inevitable event is not a random event
theorem inevitable_not_random (e : Event) :
  willDefinitelyOccur e → e ≠ Event.Random := by
  sorry

end inevitable_not_random_l3455_345591


namespace candy_challenge_solution_l3455_345589

/-- Represents the candy-eating challenge over three days -/
def candy_challenge (initial_candies : ℚ) : Prop :=
  let day1_after_eating := (3/4) * initial_candies
  let day1_remaining := day1_after_eating - 3
  let day2_after_eating := (4/5) * day1_remaining
  let day2_remaining := day2_after_eating - 5
  day2_remaining = 10

theorem candy_challenge_solution :
  ∃ (x : ℚ), candy_challenge x ∧ x = 52 :=
sorry

end candy_challenge_solution_l3455_345589


namespace mall_price_change_loss_l3455_345545

theorem mall_price_change_loss : ∀ (a b : ℝ),
  a * (1.2 : ℝ)^2 = 23.04 →
  b * (0.8 : ℝ)^2 = 23.04 →
  (a + b) - 2 * 23.04 = 5.92 := by
sorry

end mall_price_change_loss_l3455_345545


namespace quadcycle_count_l3455_345538

theorem quadcycle_count (b t q : ℕ) : 
  b + t + q = 10 →
  2*b + 3*t + 4*q = 29 →
  q = 2 :=
by sorry

end quadcycle_count_l3455_345538


namespace cone_generatrix_length_l3455_345581

/-- Represents a cone with specific properties -/
structure Cone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone
  l : ℝ  -- length of the generatrix
  lateral_area_eq : π * r * l = 2 * π * r^2  -- lateral surface area is twice the base area
  volume_eq : (1/3) * π * r^2 * h = 9 * Real.sqrt 3 * π  -- volume is 9√3π

/-- Theorem stating that a cone with the given properties has a generatrix of length 6 -/
theorem cone_generatrix_length (c : Cone) : c.l = 6 := by
  sorry

end cone_generatrix_length_l3455_345581


namespace margies_driving_distance_l3455_345539

/-- Proves that Margie can drive 400 miles with $50 worth of gas -/
theorem margies_driving_distance 
  (car_efficiency : ℝ) 
  (gas_price : ℝ) 
  (gas_budget : ℝ) 
  (h1 : car_efficiency = 40) 
  (h2 : gas_price = 5) 
  (h3 : gas_budget = 50) : 
  (gas_budget / gas_price) * car_efficiency = 400 := by
sorry

end margies_driving_distance_l3455_345539


namespace system_solutions_l3455_345521

theorem system_solutions (x y z : ℚ) : 
  ((x + 1) * (3 - 4 * y) = (6 * x + 1) * (3 - 2 * y) ∧
   (4 * x - 1) * (z + 1) = (x + 1) * (z - 1) ∧
   (3 - y) * (z - 2) = (1 - 3 * y) * (z - 6)) ↔
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨
   (x = 10/19 ∧ y = 25/7 ∧ z = 25/4)) :=
by sorry


end system_solutions_l3455_345521


namespace complex_magnitude_problem_l3455_345570

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*I)*z = -1 + 3*I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l3455_345570


namespace both_p_and_q_false_l3455_345526

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2^x > x^2

-- Define proposition q
def q : Prop := (∀ a b : ℝ, a*b > 4 → (a > 2 ∧ b > 2)) ∧ 
                ¬(∀ a b : ℝ, (a > 2 ∧ b > 2) → a*b > 4)

-- Theorem stating that both p and q are false
theorem both_p_and_q_false : ¬p ∧ ¬q := by sorry

end both_p_and_q_false_l3455_345526


namespace registration_methods_count_l3455_345529

/-- The number of subjects available for registration -/
def num_subjects : ℕ := 4

/-- The number of students registering -/
def num_students : ℕ := 3

/-- The number of different registration methods -/
def registration_methods : ℕ := num_subjects ^ num_students

/-- Theorem stating that the number of registration methods is 64 -/
theorem registration_methods_count : registration_methods = 64 := by sorry

end registration_methods_count_l3455_345529
