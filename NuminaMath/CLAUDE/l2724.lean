import Mathlib

namespace inequality_proof_l2724_272407

theorem inequality_proof (a b c d e : ℝ) 
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) : 
  a * d + d * c + c * b + b * e + e * a ≤ 1/5 := by
sorry

end inequality_proof_l2724_272407


namespace sum_lent_is_1000_l2724_272493

/-- The sum lent in rupees -/
def sum_lent : ℝ := 1000

/-- The annual interest rate as a percentage -/
def interest_rate : ℝ := 5

/-- The time period in years -/
def time_period : ℝ := 5

/-- The difference between the sum lent and the interest after the time period -/
def interest_difference : ℝ := 750

/-- Theorem stating that the sum lent is 1000 rupees given the problem conditions -/
theorem sum_lent_is_1000 :
  sum_lent = 1000 ∧
  interest_rate = 5 ∧
  time_period = 5 ∧
  interest_difference = 750 ∧
  sum_lent * interest_rate * time_period / 100 = sum_lent - interest_difference :=
by sorry

end sum_lent_is_1000_l2724_272493


namespace abs_neg_one_wrt_one_abs_wrt_one_eq_2023_l2724_272420

-- Define the absolute value with respect to 1
def abs_wrt_one (a : ℝ) : ℝ := |a - 1|

-- Theorem 1
theorem abs_neg_one_wrt_one : abs_wrt_one (-1) = 2 := by sorry

-- Theorem 2
theorem abs_wrt_one_eq_2023 (a : ℝ) : 
  abs_wrt_one a = 2023 ↔ a = 2024 ∨ a = -2022 := by sorry

end abs_neg_one_wrt_one_abs_wrt_one_eq_2023_l2724_272420


namespace fraction_equality_l2724_272431

theorem fraction_equality : (18 : ℚ) / (0.5 * 106) = 18 / 53 := by
  sorry

end fraction_equality_l2724_272431


namespace binomial_coefficient_equality_l2724_272441

theorem binomial_coefficient_equality (n s : ℕ) (h : s > 0) :
  (n.choose s) = (n * (n - 1).choose (s - 1)) / s :=
sorry

end binomial_coefficient_equality_l2724_272441


namespace emily_candy_duration_l2724_272445

/-- The number of days Emily's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  (neighbors_candy + sister_candy) / daily_consumption

/-- Theorem stating that Emily's candy will last 2 days -/
theorem emily_candy_duration :
  candy_duration 5 13 9 = 2 := by
  sorry

end emily_candy_duration_l2724_272445


namespace middle_three_sum_is_15_l2724_272451

/-- Represents a card with a color and a number. -/
inductive Card
| green (n : Nat)
| purple (n : Nat)

/-- Checks if a stack of cards satisfies the given conditions. -/
def validStack (stack : List Card) : Prop :=
  let greenCards := [1, 2, 3, 4, 5, 6]
  let purpleCards := [4, 5, 6, 7, 8]
  (∀ c ∈ stack, match c with
    | Card.green n => n ∈ greenCards
    | Card.purple n => n ∈ purpleCards) ∧
  (stack.length = 11) ∧
  (∀ i, i % 2 = 0 → match stack.get? i with
    | some (Card.green _) => True
    | _ => False) ∧
  (∀ i, i % 2 = 1 → match stack.get? i with
    | some (Card.purple _) => True
    | _ => False) ∧
  (∀ i, i + 1 < stack.length →
    match stack.get? i, stack.get? (i + 1) with
    | some (Card.green m), some (Card.purple n) => n % m = 0 ∧ n > m
    | some (Card.purple n), some (Card.green m) => n % m = 0 ∧ n > m
    | _, _ => True)

/-- The sum of the numbers on the middle three cards in a valid stack is 15. -/
theorem middle_three_sum_is_15 (stack : List Card) :
  validStack stack →
  (match stack.get? 4, stack.get? 5, stack.get? 6 with
   | some (Card.purple n1), some (Card.green n2), some (Card.purple n3) =>
     n1 + n2 + n3 = 15
   | _, _, _ => False) :=
by sorry

end middle_three_sum_is_15_l2724_272451


namespace total_employees_is_100_l2724_272471

/-- The ratio of employees in groups A, B, and C -/
def group_ratio : Fin 3 → ℕ
  | 0 => 5  -- Group A
  | 1 => 4  -- Group B
  | 2 => 1  -- Group C

/-- The total sample size -/
def sample_size : ℕ := 20

/-- The probability of selecting both person A and person B in group C -/
def prob_select_two : ℚ := 1 / 45

theorem total_employees_is_100 :
  ∀ (total : ℕ),
  (∃ (group_C_size : ℕ),
    /- Group C size is 1/10 of the total -/
    group_C_size = total / 10 ∧
    /- The probability of selecting 2 from group C matches the given probability -/
    (group_C_size.choose 2 : ℚ) / total.choose 2 = prob_select_two ∧
    /- The sample size for group C is 2 -/
    group_C_size * sample_size / total = 2) →
  total = 100 := by
sorry

end total_employees_is_100_l2724_272471


namespace fraction_equality_l2724_272414

theorem fraction_equality (a b c : ℝ) (h1 : c ≠ 0) :
  a / c = b / c → a = b := by sorry

end fraction_equality_l2724_272414


namespace no_solution_exists_l2724_272472

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the properties of the function f
def StrictlyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- State the theorem
theorem no_solution_exists (f : ℝ → ℝ) 
  (h1 : StrictlyDecreasing f)
  (h2 : ∀ x ∈ PositiveReals, f x * f (f x + 3 / (2 * x)) = 1/4) :
  ¬ ∃ x ∈ PositiveReals, f x + 3 * x = 2 := by
  sorry

end no_solution_exists_l2724_272472


namespace repeating_decimal_equals_three_elevenths_l2724_272461

/-- The repeating decimal 0.27̄ -/
def repeating_decimal : ℚ := 27 / 99

theorem repeating_decimal_equals_three_elevenths : 
  repeating_decimal = 3 / 11 := by sorry

end repeating_decimal_equals_three_elevenths_l2724_272461


namespace specific_competition_scores_l2724_272418

/-- Represents a mathematics competition with the given scoring rules. -/
structure MathCompetition where
  total_problems : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  unattempted_points : ℤ

/-- Calculates the number of different total scores possible in the competition. -/
def countPossibleScores (comp : MathCompetition) : ℕ := sorry

/-- The specific competition described in the problem. -/
def specificCompetition : MathCompetition :=
  { total_problems := 30
  , correct_points := 4
  , incorrect_points := -1
  , unattempted_points := 0 }

/-- Theorem stating that the number of different total scores in the specific competition is 145. -/
theorem specific_competition_scores :
  countPossibleScores specificCompetition = 145 := by sorry

end specific_competition_scores_l2724_272418


namespace cubic_function_property_l2724_272442

/-- Given a cubic function f(x) = ax³ - bx + 5 where a and b are real numbers,
    if f(-3) = -1, then f(3) = 11. -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^3 - b * x + 5)
    (h2 : f (-3) = -1) : f 3 = 11 := by
  sorry

end cubic_function_property_l2724_272442


namespace f_zero_f_odd_f_range_l2724_272410

noncomputable section

variable (f : ℝ → ℝ)

-- Define the properties of f
axiom add_hom : ∀ x y : ℝ, f (x + y) = f x + f y
axiom pos_map_pos : ∀ x : ℝ, x > 0 → f x > 0
axiom f_neg_one : f (-1) = -2

-- Theorem statements
theorem f_zero : f 0 = 0 := by sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem f_range : ∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f x ∈ Set.Icc (-4) 2 := by sorry

end

end f_zero_f_odd_f_range_l2724_272410


namespace billy_basketball_points_difference_l2724_272415

theorem billy_basketball_points_difference : 
  ∀ (billy_points friend_points : ℕ),
    billy_points = 7 →
    friend_points = 9 →
    friend_points - billy_points = 2 := by
  sorry

end billy_basketball_points_difference_l2724_272415


namespace moose_population_canada_l2724_272454

/-- The moose population in Canada, given the ratio of moose to beavers to humans -/
theorem moose_population_canada (total_humans : ℕ) (moose_to_beaver : ℕ) (beaver_to_human : ℕ) :
  total_humans = 38000000 →
  moose_to_beaver = 2 →
  beaver_to_human = 19 →
  (total_humans / (moose_to_beaver * beaver_to_human) : ℚ) = 1000000 := by
  sorry

#check moose_population_canada

end moose_population_canada_l2724_272454


namespace min_a_for_increasing_cubic_l2724_272421

/-- Given a function f(x) = x^3 + ax that is increasing on [1, +∞), 
    the minimum value of a is -3. -/
theorem min_a_for_increasing_cubic (a : ℝ) : 
  (∀ x ≥ 1, Monotone (fun x => x^3 + a*x)) → a ≥ -3 := by
  sorry

end min_a_for_increasing_cubic_l2724_272421


namespace fish_count_l2724_272449

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 12

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 22 := by
  sorry

end fish_count_l2724_272449


namespace melted_ice_cream_depth_l2724_272419

/-- Given a sphere of ice cream that melts into a cylinder, calculate the height of the cylinder -/
theorem melted_ice_cream_depth (initial_radius final_radius : ℝ) 
  (initial_radius_pos : 0 < initial_radius)
  (final_radius_pos : 0 < final_radius)
  (h_initial_radius : initial_radius = 3)
  (h_final_radius : final_radius = 12) :
  (4 / 3 * Real.pi * initial_radius ^ 3) / (Real.pi * final_radius ^ 2) = 1 / 4 := by
  sorry

end melted_ice_cream_depth_l2724_272419


namespace cube_surface_area_l2724_272413

/-- The surface area of a cube with edge length 11 cm is 726 cm². -/
theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 11) :
  6 * edge_length^2 = 726 := by
  sorry

end cube_surface_area_l2724_272413


namespace geometric_sequence_sum_l2724_272404

theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) :
  S n = 48 ∧ S (2 * n) = 60 →
  S (3 * n) = 63 := by
sorry

end geometric_sequence_sum_l2724_272404


namespace perpendicular_vectors_m_value_l2724_272465

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 1]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem perpendicular_vectors_m_value :
  ∀ m : ℝ, dot_product a (b m) = 0 → m = -2 := by
  sorry

end perpendicular_vectors_m_value_l2724_272465


namespace r_fourth_plus_inverse_r_fourth_l2724_272485

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
  sorry

end r_fourth_plus_inverse_r_fourth_l2724_272485


namespace odds_calculation_l2724_272460

theorem odds_calculation (x : ℝ) (h : (x / (x + 5)) = 0.375) : x = 3 := by
  sorry

end odds_calculation_l2724_272460


namespace existence_of_nine_digit_combination_l2724_272427

theorem existence_of_nine_digit_combination : ∃ (a b c d e f g h i : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 100 ∧
  (a + b + c + d + e + f + g + h * i = 100 ∨
   a + b + c + d + e * f + g + h = 100 ∨
   a + b + c + d + e - f - g + h + i = 100) :=
by sorry

end existence_of_nine_digit_combination_l2724_272427


namespace unvisited_route_count_l2724_272462

/-- The number of ways to distribute four families among four routes with one route unvisited -/
def unvisited_route_scenarios : ℕ := 144

/-- The number of families -/
def num_families : ℕ := 4

/-- The number of available routes -/
def num_routes : ℕ := 4

theorem unvisited_route_count :
  unvisited_route_scenarios = 
    (Nat.choose num_families 2) * (Nat.factorial num_routes) / Nat.factorial (num_routes - 3) :=
sorry

end unvisited_route_count_l2724_272462


namespace gcd_of_powers_of_two_l2724_272443

theorem gcd_of_powers_of_two : Nat.gcd (2^1005 - 1) (2^1016 - 1) = 2^11 - 1 := by
  sorry

end gcd_of_powers_of_two_l2724_272443


namespace smallest_k_for_difference_property_l2724_272401

theorem smallest_k_for_difference_property (n : ℕ) (hn : n ≥ 1) :
  let k := n^2 + 2
  ∀ (S : Finset ℝ), S.card ≥ k →
    ∃ (x y : ℝ), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧
      (|x - y| < 1 / n ∨ |x - y| > n) ∧
    ∀ (m : ℕ), m < k →
      ∃ (T : Finset ℝ), T.card = m ∧
        ∀ (a b : ℝ), a ∈ T ∧ b ∈ T ∧ a ≠ b →
          |a - b| ≥ 1 / n ∧ |a - b| ≤ n :=
sorry

end smallest_k_for_difference_property_l2724_272401


namespace count_numbers_with_at_most_two_digits_is_2034_l2724_272491

/-- Counts the number of positive integers less than 100000 with at most two different digits. -/
def count_numbers_with_at_most_two_digits : ℕ :=
  let single_digit_count : ℕ := 45
  let two_digits_with_zero_count : ℕ := 117
  let two_digits_without_zero_count : ℕ := 1872
  single_digit_count + two_digits_with_zero_count + two_digits_without_zero_count

/-- The count of positive integers less than 100000 with at most two different digits is 2034. -/
theorem count_numbers_with_at_most_two_digits_is_2034 :
  count_numbers_with_at_most_two_digits = 2034 := by
  sorry

end count_numbers_with_at_most_two_digits_is_2034_l2724_272491


namespace hyperbola_intersection_theorem_l2724_272455

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the right branch of the hyperbola -/
def isOnRightBranch (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1 ∧ p.x > 0

/-- Checks if a point is on the given line -/
def isOnLine (p : Point) : Prop :=
  p.y = Real.sqrt 3 / 3 * p.x - 2

/-- The main theorem to be proved -/
theorem hyperbola_intersection_theorem (h : Hyperbola) 
    (hA : isOnRightBranch h A ∧ isOnLine A)
    (hB : isOnRightBranch h B ∧ isOnLine B)
    (hC : isOnRightBranch h C) :
    h.a = 2 * Real.sqrt 3 →
    h.b = Real.sqrt 3 →
    C.x = 4 * Real.sqrt 3 →
    C.y = 3 →
    ∃ m : ℝ, m = 4 ∧ A.x + B.x = m * C.x ∧ A.y + B.y = m * C.y := by
  sorry


end hyperbola_intersection_theorem_l2724_272455


namespace jeans_business_weekly_hours_l2724_272405

/-- Represents the operating hours of a business for a single day -/
structure DailyHours where
  open_time : Nat
  close_time : Nat

/-- Calculates the number of hours a business is open in a day -/
def hours_open (dh : DailyHours) : Nat :=
  dh.close_time - dh.open_time

/-- Represents the operating hours of Jean's business for a week -/
structure WeeklyHours where
  weekday_hours : DailyHours
  weekend_hours : DailyHours

/-- Calculates the total number of hours Jean's business is open in a week -/
def total_weekly_hours (wh : WeeklyHours) : Nat :=
  (hours_open wh.weekday_hours * 5) + (hours_open wh.weekend_hours * 2)

/-- Jean's business hours -/
def jeans_business : WeeklyHours :=
  { weekday_hours := { open_time := 16, close_time := 22 }
  , weekend_hours := { open_time := 18, close_time := 22 } }

theorem jeans_business_weekly_hours :
  total_weekly_hours jeans_business = 38 := by
  sorry

end jeans_business_weekly_hours_l2724_272405


namespace delores_remaining_money_l2724_272489

/-- Calculates the remaining money after purchases -/
def remaining_money (initial : ℕ) (computer : ℕ) (printer : ℕ) : ℕ :=
  initial - (computer + printer)

/-- Proves that Delores has $10 left after her purchases -/
theorem delores_remaining_money :
  remaining_money 450 400 40 = 10 := by
  sorry

end delores_remaining_money_l2724_272489


namespace crayon_boxes_l2724_272487

theorem crayon_boxes (total_crayons : ℕ) (full_boxes : ℕ) (loose_crayons : ℕ) (friend_crayons : ℕ) :
  total_crayons = 85 →
  full_boxes = 5 →
  loose_crayons = 5 →
  friend_crayons = 27 →
  (total_crayons - loose_crayons) / full_boxes = 16 →
  ((loose_crayons + friend_crayons) + ((total_crayons - loose_crayons) / full_boxes - 1)) / ((total_crayons - loose_crayons) / full_boxes) = 2 := by
  sorry

#check crayon_boxes

end crayon_boxes_l2724_272487


namespace solve_linear_equation_l2724_272473

theorem solve_linear_equation (x : ℚ) (h : x + 2*x + 3*x + 4*x = 5) : x = 1/2 := by
  sorry

end solve_linear_equation_l2724_272473


namespace arithmetic_geometric_sequence_formula_l2724_272453

/-- An arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ n, a (n + 1) = a n * q

/-- The general term formula for the sequence -/
def GeneralTerm (a : ℕ → ℝ) : Prop :=
  (∃ (c : ℝ), ∀ n, a n = 125 * (2/5)^(n-1)) ∨
  (∃ (c : ℝ), ∀ n, a n = 8 * (5/2)^(n-1))

theorem arithmetic_geometric_sequence_formula (a : ℕ → ℝ) 
  (h1 : ArithGeomSeq a)
  (h2 : a 1 + a 4 = 133)
  (h3 : a 2 + a 3 = 70) :
  GeneralTerm a :=
sorry

end arithmetic_geometric_sequence_formula_l2724_272453


namespace lighter_person_weight_l2724_272480

/-- Given two people with a total weight of 88 kg, where one person is 4 kg heavier than the other,
    prove that the weight of the lighter person is 42 kg. -/
theorem lighter_person_weight (total_weight : ℝ) (weight_difference : ℝ) (lighter_weight : ℝ) : 
  total_weight = 88 → weight_difference = 4 → 
  lighter_weight + (lighter_weight + weight_difference) = total_weight →
  lighter_weight = 42 := by
  sorry

#check lighter_person_weight

end lighter_person_weight_l2724_272480


namespace line_slope_value_l2724_272432

/-- Given a line l passing through points A(3, m+1) and B(4, 2m+1) with slope π/4, prove that m = 1 -/
theorem line_slope_value (m : ℝ) : 
  (∃ l : Set (ℝ × ℝ), 
    (3, m + 1) ∈ l ∧ 
    (4, 2*m + 1) ∈ l ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = Real.pi / 4)) →
  m = 1 := by
sorry

end line_slope_value_l2724_272432


namespace sum_cube_value_l2724_272428

theorem sum_cube_value (x y : ℝ) (h1 : x * (x + y) = 49) (h2 : y * (x + y) = 63) :
  (x + y)^3 = 448 * Real.sqrt 7 := by
sorry

end sum_cube_value_l2724_272428


namespace perpendicular_lines_a_value_l2724_272409

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y - 1 = 0 ↔ 2*x - 4*y + 5 = 0) → a = 4 := by
  sorry

end perpendicular_lines_a_value_l2724_272409


namespace bus_capacity_theorem_l2724_272470

/-- Represents the capacity of a bus in terms of children -/
def bus_capacity (rows : ℕ) (children_per_row : ℕ) : ℕ :=
  rows * children_per_row

/-- Theorem stating that a bus with 9 rows and 4 children per row can accommodate 36 children -/
theorem bus_capacity_theorem :
  bus_capacity 9 4 = 36 := by
  sorry

end bus_capacity_theorem_l2724_272470


namespace parabola_directrix_l2724_272438

/-- 
Given a parabola y² = 2px with intersection point (4, 0), 
prove that its directrix has the equation x = -4 
-/
theorem parabola_directrix (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Equation of the parabola
  (0^2 = 2*p*4) →            -- Intersection point (4, 0)
  (x = -4) →                 -- Equation of the directrix
  True := by sorry

end parabola_directrix_l2724_272438


namespace circle_equation_with_tangent_line_l2724_272484

/-- The equation of a circle with center (a, b) and radius r is (x-a)^2 + (y-b)^2 = r^2 -/
def CircleEquation (a b r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

/-- A line is tangent to a circle if the distance from the center of the circle to the line equals the radius of the circle -/
def IsTangentLine (a b r : ℝ) (m n c : ℝ) : Prop :=
  r = |m*a + n*b + c| / Real.sqrt (m^2 + n^2)

/-- The theorem stating that (x-2)^2 + (y+1)^2 = 8 is the equation of the circle with center (2, -1) tangent to the line x + y = 5 -/
theorem circle_equation_with_tangent_line :
  ∀ x y : ℝ,
  CircleEquation 2 (-1) (Real.sqrt 8) x y ↔ 
  IsTangentLine 2 (-1) (Real.sqrt 8) 1 1 (-5) :=
by sorry

end circle_equation_with_tangent_line_l2724_272484


namespace negation_of_universal_statement_l2724_272496

theorem negation_of_universal_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, 3 * x - 5 > 0) ↔ (∃ x ∈ S, 3 * x - 5 ≤ 0) := by
  sorry

end negation_of_universal_statement_l2724_272496


namespace geometric_progression_and_quadratic_vertex_l2724_272490

/-- Given that a, b, c, d are in geometric progression and the vertex of y = x^2 - 2x + 3 is (b, c),
    prove that a + d = 9/2 -/
theorem geometric_progression_and_quadratic_vertex 
  (a b c d : ℝ) 
  (h1 : ∃ (q : ℝ), q ≠ 0 ∧ b = a * q ∧ c = b * q ∧ d = c * q) 
  (h2 : b^2 - 2*b + 3 = c) : 
  a + d = 9/2 := by sorry

end geometric_progression_and_quadratic_vertex_l2724_272490


namespace fraction_addition_l2724_272477

theorem fraction_addition (c : ℝ) : (4 + 3 * c) / 7 + 2 = (18 + 3 * c) / 7 := by
  sorry

end fraction_addition_l2724_272477


namespace fourth_place_points_value_l2724_272411

def first_place_points : ℕ := 11
def second_place_points : ℕ := 7
def third_place_points : ℕ := 5
def total_participations : ℕ := 7
def total_points_product : ℕ := 38500

def is_valid_fourth_place_points (fourth_place_points : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    a + b + c + d = total_participations ∧
    first_place_points ^ a * second_place_points ^ b * third_place_points ^ c * fourth_place_points ^ d = total_points_product

theorem fourth_place_points_value :
  ∃! (x : ℕ), is_valid_fourth_place_points x ∧ x = 4 := by sorry

end fourth_place_points_value_l2724_272411


namespace line_circle_no_intersection_l2724_272488

/-- The line 3x + 4y = 12 and the circle x^2 + y^2 = 4 have no intersection points in the real plane. -/
theorem line_circle_no_intersection :
  ¬ ∃ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) := by
  sorry

end line_circle_no_intersection_l2724_272488


namespace b_10_equals_64_l2724_272469

/-- Sequences a and b satisfying the given conditions -/
def sequences_a_b (a b : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  ∀ n : ℕ, (a n) * (a (n + 1)) = 2^n ∧
           (a n) + (a (n + 1)) = b n

/-- The main theorem to prove -/
theorem b_10_equals_64 (a b : ℕ → ℝ) (h : sequences_a_b a b) : 
  b 10 = 64 := by
  sorry

end b_10_equals_64_l2724_272469


namespace angle_measure_from_coordinates_l2724_272403

/-- Given an acute angle α and a point A on its terminal side with coordinates (2sin 3, -2cos 3),
    prove that α = 3 - π/2 --/
theorem angle_measure_from_coordinates (α : Real) (A : Real × Real) :
  α > 0 ∧ α < π/2 →  -- α is acute
  A.1 = 2 * Real.sin 3 ∧ A.2 = -2 * Real.cos 3 →  -- coordinates of A
  α = 3 - π/2 := by
  sorry

end angle_measure_from_coordinates_l2724_272403


namespace max_value_of_function_max_value_achieved_l2724_272439

theorem max_value_of_function (x : ℝ) (h : x < 0) : x + 4/x ≤ -4 := by
  sorry

theorem max_value_achieved (x : ℝ) (h : x < 0) : ∃ x₀, x₀ < 0 ∧ x₀ + 4/x₀ = -4 := by
  sorry

end max_value_of_function_max_value_achieved_l2724_272439


namespace line_up_arrangement_count_l2724_272444

/-- The number of different arrangements of 5 students (2 boys and 3 girls) where only two girls are adjacent. -/
def arrangement_count : ℕ := 24

/-- The total number of students in the line-up. -/
def total_students : ℕ := 5

/-- The number of boys in the line-up. -/
def num_boys : ℕ := 2

/-- The number of girls in the line-up. -/
def num_girls : ℕ := 3

/-- The number of adjacent girls in each arrangement. -/
def adjacent_girls : ℕ := 2

theorem line_up_arrangement_count :
  arrangement_count = 24 ∧
  total_students = num_boys + num_girls ∧
  num_boys = 2 ∧
  num_girls = 3 ∧
  adjacent_girls = 2 :=
sorry

end line_up_arrangement_count_l2724_272444


namespace floor_sqrt_80_l2724_272450

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end floor_sqrt_80_l2724_272450


namespace solution_system_equations_l2724_272475

theorem solution_system_equations (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_order : a > b ∧ b > c ∧ c > d) :
  ∃ (x y z t : ℝ),
    (|a - b| * y + |a - c| * z + |a - d| * t = 1) ∧
    (|b - a| * x + |b - c| * z + |b - d| * t = 1) ∧
    (|c - a| * x + |c - b| * y + |c - d| * t = 1) ∧
    (|d - a| * x + |d - b| * y + |d - c| * z = 1) ∧
    (x = 1 / (a - d)) ∧
    (y = 0) ∧
    (z = 0) ∧
    (t = 1 / (a - d)) := by
  sorry

end solution_system_equations_l2724_272475


namespace daryl_crate_loading_problem_l2724_272423

theorem daryl_crate_loading_problem :
  let crate_capacity : ℕ := 20
  let num_crates : ℕ := 15
  let total_capacity : ℕ := crate_capacity * num_crates
  let num_nail_bags : ℕ := 4
  let nail_bag_weight : ℕ := 5
  let num_hammer_bags : ℕ := 12
  let hammer_bag_weight : ℕ := 5
  let num_plank_bags : ℕ := 10
  let plank_bag_weight : ℕ := 30
  let total_nail_weight : ℕ := num_nail_bags * nail_bag_weight
  let total_hammer_weight : ℕ := num_hammer_bags * hammer_bag_weight
  let total_plank_weight : ℕ := num_plank_bags * plank_bag_weight
  let total_item_weight : ℕ := total_nail_weight + total_hammer_weight + total_plank_weight
  total_item_weight - total_capacity = 80 :=
by sorry

end daryl_crate_loading_problem_l2724_272423


namespace regular_polygon_sides_l2724_272440

/-- A regular polygon with perimeter 49 and side length 7 has 7 sides. -/
theorem regular_polygon_sides (p : ℕ) (s : ℕ) (h1 : p = 49) (h2 : s = 7) :
  p / s = 7 := by
  sorry

end regular_polygon_sides_l2724_272440


namespace both_false_sufficient_not_necessary_l2724_272458

-- Define simple propositions a and b
variable (a b : Prop)

-- Define the statements
def both_false : Prop := ¬a ∧ ¬b
def either_false : Prop := ¬a ∨ ¬b

-- Theorem statement
theorem both_false_sufficient_not_necessary :
  (both_false a b → either_false a b) ∧
  ¬(either_false a b → both_false a b) :=
sorry

end both_false_sufficient_not_necessary_l2724_272458


namespace smallest_four_digit_mod_8_3_l2724_272481

theorem smallest_four_digit_mod_8_3 : ∀ n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧ n % 8 = 3 → n ≥ 1003 :=
by sorry

end smallest_four_digit_mod_8_3_l2724_272481


namespace max_necklaces_proof_l2724_272429

def necklace_green_beads : ℕ := 9
def necklace_white_beads : ℕ := 6
def necklace_orange_beads : ℕ := 3
def available_beads : ℕ := 45

def max_necklaces : ℕ := 5

theorem max_necklaces_proof :
  min (available_beads / necklace_green_beads)
      (min (available_beads / necklace_white_beads)
           (available_beads / necklace_orange_beads)) = max_necklaces := by
  sorry

end max_necklaces_proof_l2724_272429


namespace quadratic_inequality_solution_quadratic_inequality_real_solution_l2724_272417

/-- The quadratic inequality -/
def quadratic_inequality (k x : ℝ) : Prop := k * x^2 - 2 * x + 6 * k < 0

/-- The solution set for part 1 -/
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2

/-- The solution set for part 2 -/
def solution_set_2 : Set ℝ := Set.univ

theorem quadratic_inequality_solution (k : ℝ) :
  (k ≠ 0 ∧ ∀ x, quadratic_inequality k x ↔ solution_set_1 x) → k = -2/5 :=
sorry

theorem quadratic_inequality_real_solution (k : ℝ) :
  (k ≠ 0 ∧ ∀ x, quadratic_inequality k x ↔ x ∈ solution_set_2) → 
  k < 0 ∧ k < -Real.sqrt 6 / 6 :=
sorry

end quadratic_inequality_solution_quadratic_inequality_real_solution_l2724_272417


namespace polynomial_remainder_l2724_272466

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 7*x^3 + 9*x^2 + 16*x - 13
  f 3 = 8 := by sorry

end polynomial_remainder_l2724_272466


namespace max_value_of_f_l2724_272402

def f (x : ℝ) : ℝ := -3 * x^2 + 8

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x₀ : ℝ), f x₀ = M) ∧ M = 8 :=
sorry

end max_value_of_f_l2724_272402


namespace unique_solution_for_odd_prime_l2724_272446

theorem unique_solution_for_odd_prime (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), 
    (2 : ℚ) / p = 1 / m + 1 / n ∧
    m > n ∧
    m = p * (p + 1) / 2 ∧
    n = 2 / (p + 1) := by
  sorry

end unique_solution_for_odd_prime_l2724_272446


namespace max_value_a_l2724_272459

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 150) : 
  a ≤ 8924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 8924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 150 :=
sorry

end max_value_a_l2724_272459


namespace division_remainder_l2724_272435

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 760 →
  divisor = 36 →
  quotient = 21 →
  dividend = divisor * quotient + remainder →
  remainder = 4 := by
sorry

end division_remainder_l2724_272435


namespace moving_circle_theorem_l2724_272474

-- Define the circle and its properties
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_point : center.1^2 + center.2^2 = (center.1 - 1)^2 + center.2^2
  tangent_to_line : (center.1 + 1)^2 = center.1^2 + center.2^2

-- Define the trajectory
def trajectory (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the condition for two points on the trajectory
def trajectory_points_condition (A B : ℝ × ℝ) : Prop :=
  trajectory A ∧ trajectory B ∧ A ≠ B ∧
  (A.2 / A.1) * (B.2 / B.1) = 1

-- The main theorem
theorem moving_circle_theorem (C : MovingCircle) :
  (∀ p : ℝ × ℝ, p = C.center → trajectory p) ∧
  (∀ A B : ℝ × ℝ, trajectory_points_condition A B →
    ∃ k : ℝ, B.2 - A.2 = k * (B.1 - A.1) ∧ A.2 = k * (A.1 + 4)) :=
sorry

end moving_circle_theorem_l2724_272474


namespace range_of_m_for_nonempty_solution_l2724_272400

theorem range_of_m_for_nonempty_solution (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → -5 ≤ m ∧ m ≤ 3 := by
  sorry

end range_of_m_for_nonempty_solution_l2724_272400


namespace max_sum_xyz_l2724_272468

theorem max_sum_xyz (x y z : ℕ+) (h1 : x < y) (h2 : y < z) (h3 : x + x * y + x * y * z = 37) :
  x + y + z ≤ 20 :=
sorry

end max_sum_xyz_l2724_272468


namespace parallelogram_area_l2724_272476

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : θ = π/4) :
  a * b * Real.sin θ = 6 * Real.sqrt 2 := by
  sorry

end parallelogram_area_l2724_272476


namespace new_students_average_age_l2724_272452

theorem new_students_average_age
  (initial_count : ℕ)
  (initial_avg : ℝ)
  (new_count : ℕ)
  (new_avg_increase : ℝ)
  (h1 : initial_count = 10)
  (h2 : initial_avg = 14)
  (h3 : new_count = 5)
  (h4 : new_avg_increase = 1) :
  let total_initial := initial_count * initial_avg
  let total_new := (initial_count + new_count) * (initial_avg + new_avg_increase)
  let new_students_total := total_new - total_initial
  new_students_total / new_count = 17 := by
sorry

end new_students_average_age_l2724_272452


namespace min_value_of_f_l2724_272433

/-- The function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := (x^2 - 1) * (x^2 + a*x + b)

/-- The theorem stating the minimum value of f(x) -/
theorem min_value_of_f (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (4 - x)) →
  (∃ x₀ : ℝ, ∀ x : ℝ, f a b x₀ ≤ f a b x) ∧
  (∃ x₁ : ℝ, f a b x₁ = -16) :=
by sorry

end min_value_of_f_l2724_272433


namespace problem_1_problem_2_l2724_272467

theorem problem_1 : (1 - Real.sqrt 2) ^ 0 - 2 * Real.sin (π / 4) + (Real.sqrt 2) ^ 2 = 3 - Real.sqrt 2 := by
  sorry

theorem problem_2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) :
  (1 - 2 / (x - 1)) / ((x^2 - 6*x + 9) / (x^2 - 1)) = (x + 1) / (x - 3) := by
  sorry

end problem_1_problem_2_l2724_272467


namespace abs_minus_2010_l2724_272448

theorem abs_minus_2010 : |(-2010 : ℤ)| = 2010 := by
  sorry

end abs_minus_2010_l2724_272448


namespace third_box_weight_l2724_272456

/-- Given two boxes with weights and their weight difference, prove the weight of the third box -/
theorem third_box_weight (weight_first : ℝ) (weight_diff : ℝ) : 
  weight_first = 2 → weight_diff = 11 → weight_first + weight_diff = 13 := by
  sorry

end third_box_weight_l2724_272456


namespace distribute_five_contestants_three_companies_l2724_272497

/-- The number of ways to distribute contestants among companies -/
def distribute_contestants (num_contestants : ℕ) (num_companies : ℕ) : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem: The number of ways to distribute 5 contestants among 3 companies,
    where each company must have at least 1 and at most 2 contestants, is 90 -/
theorem distribute_five_contestants_three_companies :
  distribute_contestants 5 3 = 90 := by
  sorry

end distribute_five_contestants_three_companies_l2724_272497


namespace investment_income_calculation_l2724_272436

/-- Calculates the annual income from an investment in shares given the investment amount,
    share face value, quoted price, and dividend rate. -/
def annual_income (investment : ℚ) (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) : ℚ :=
  (investment / quoted_price) * (face_value * dividend_rate)

/-- Theorem stating that for the given investment scenario, the annual income is 728 -/
theorem investment_income_calculation :
  let investment : ℚ := 4940
  let face_value : ℚ := 10
  let quoted_price : ℚ := 9.5
  let dividend_rate : ℚ := 14 / 100
  annual_income investment face_value quoted_price dividend_rate = 728 := by
  sorry


end investment_income_calculation_l2724_272436


namespace equation_solution_l2724_272425

theorem equation_solution : ∃! x : ℝ, 45 - (28 - (37 - (x - 17))) = 56 ∧ x = 15 := by
  sorry

end equation_solution_l2724_272425


namespace arithmetic_sequence_sum_2017_l2724_272492

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_2017 
  (seq : ArithmeticSequence) 
  (h1 : sum_n seq 2011 = -2011) 
  (h2 : seq.a 1012 = 3) : 
  sum_n seq 2017 = 2017 := by
  sorry

end arithmetic_sequence_sum_2017_l2724_272492


namespace fraction_to_decimal_l2724_272494

theorem fraction_to_decimal (n : ℕ) (d : ℕ) (h : d = 5^4 * 2) :
  (n : ℚ) / d = 47 / d → (n : ℚ) / d = 0.0376 := by
  sorry

end fraction_to_decimal_l2724_272494


namespace smallest_dual_base_representation_l2724_272416

/-- Represents a digit in a given base --/
def IsDigit (d : ℕ) (base : ℕ) : Prop := d < base

/-- Converts a two-digit number in a given base to base 10 --/
def ToBase10 (d : ℕ) (base : ℕ) : ℕ := base * d + d

/-- The problem statement --/
theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (c d : ℕ),
    IsDigit c 6 ∧
    IsDigit d 8 ∧
    ToBase10 c 6 = n ∧
    ToBase10 d 8 = n ∧
    (∀ (m : ℕ) (c' d' : ℕ),
      IsDigit c' 6 →
      IsDigit d' 8 →
      ToBase10 c' 6 = m →
      ToBase10 d' 8 = m →
      n ≤ m) ∧
    n = 63 := by
  sorry

end smallest_dual_base_representation_l2724_272416


namespace number_of_divisors_3960_l2724_272495

theorem number_of_divisors_3960 : Nat.card (Nat.divisors 3960) = 48 := by
  sorry

end number_of_divisors_3960_l2724_272495


namespace xyz_value_l2724_272463

theorem xyz_value (a b c x y z : ℂ) 
  (eq1 : a = (b + c) / (x - 2))
  (eq2 : b = (c + a) / (y - 2))
  (eq3 : c = (a + b) / (z - 2))
  (sum_prod : x * y + y * z + z * x = 67)
  (sum : x + y + z = 2010) :
  x * y * z = -5892 := by
sorry

end xyz_value_l2724_272463


namespace john_money_left_l2724_272434

/-- Proves that John has $65 left after giving money to his parents -/
theorem john_money_left (initial_amount : ℚ) : 
  initial_amount = 200 →
  initial_amount - (3/8 * initial_amount + 3/10 * initial_amount) = 65 := by
  sorry

end john_money_left_l2724_272434


namespace parkway_elementary_soccer_l2724_272437

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (h1 : total_students = 450)
  (h2 : boys = 320)
  (h3 : soccer_players = 250)
  (h4 : (86 : ℚ) / 100 * soccer_players = ↑(boys_playing_soccer))
  (boys_playing_soccer : ℕ) :
  total_students - boys - (soccer_players - boys_playing_soccer) = 95 :=
by sorry

end parkway_elementary_soccer_l2724_272437


namespace equation_solution_l2724_272478

theorem equation_solution : 
  ∀ x : ℝ, x > 0 → (x^(Real.log x / Real.log 10) = x^5 / 10000 ↔ x = 10 ∨ x = 10000) :=
by sorry

end equation_solution_l2724_272478


namespace number_of_divisors_of_fourth_power_l2724_272408

/-- Given a positive integer n where n = p₁ * p₂² * p₃⁵ and p₁, p₂, and p₃ are different prime numbers,
    the number of positive divisors of x = n⁴ is 945. -/
theorem number_of_divisors_of_fourth_power (p₁ p₂ p₃ : Nat) (h_prime₁ : Prime p₁) (h_prime₂ : Prime p₂)
    (h_prime₃ : Prime p₃) (h_distinct : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃) :
    let n := p₁ * p₂^2 * p₃^5
    let x := n^4
    (Nat.divisors x).card = 945 := by
  sorry

#check number_of_divisors_of_fourth_power

end number_of_divisors_of_fourth_power_l2724_272408


namespace regular_octagon_interior_angle_l2724_272483

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ := by
  -- Define the number of sides in an octagon
  let n : ℕ := 8

  -- Define the sum of interior angles for an n-sided polygon
  let sum_interior_angles : ℝ := 180 * (n - 2)

  -- Define the measure of one interior angle
  let one_angle : ℝ := sum_interior_angles / n

  -- Prove that one_angle equals 135
  sorry

end regular_octagon_interior_angle_l2724_272483


namespace range_of_a_l2724_272424

/-- Curve C1 in polar coordinates -/
def C1 (ρ θ a : ℝ) : Prop := ρ * (Real.sqrt 2 * Real.cos θ - Real.sin θ) = a

/-- Curve C2 in parametric form -/
def C2 (x y θ : ℝ) : Prop := x = Real.sin θ + Real.cos θ ∧ y = 1 + Real.sin (2 * θ)

/-- C1 in rectangular coordinates -/
def C1_rect (x y a : ℝ) : Prop := Real.sqrt 2 * x - y - a = 0

/-- C2 in rectangular coordinates -/
def C2_rect (x y : ℝ) : Prop := y = x^2 ∧ x ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)

/-- The main theorem -/
theorem range_of_a (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    C1_rect x₁ y₁ a ∧ C2_rect x₁ y₁ ∧
    C1_rect x₂ y₂ a ∧ C2_rect x₂ y₂) ↔
  a ∈ Set.Icc (-1/2) 4 :=
sorry

end range_of_a_l2724_272424


namespace intersection_coordinate_product_prove_intersection_coordinate_product_l2724_272486

/-- The product of coordinates of intersection points of two specific circles is 8 -/
theorem intersection_coordinate_product : ℝ → Prop := fun r =>
  ∀ x y : ℝ,
  (x^2 - 4*x + y^2 - 8*y + 20 = 0 ∧ x^2 - 6*x + y^2 - 8*y + 25 = 0) →
  r = x * y ∧ r = 8

/-- Proof of the theorem -/
theorem prove_intersection_coordinate_product :
  ∃ r : ℝ, intersection_coordinate_product r :=
by
  sorry

end intersection_coordinate_product_prove_intersection_coordinate_product_l2724_272486


namespace largest_prime_divisor_of_sum_of_squares_l2724_272499

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (36^2 + 45^2) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (36^2 + 45^2) → q ≤ p :=
by sorry

end largest_prime_divisor_of_sum_of_squares_l2724_272499


namespace solution_to_sqrt_equation_l2724_272464

theorem solution_to_sqrt_equation :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
sorry

end solution_to_sqrt_equation_l2724_272464


namespace sandy_scooter_price_l2724_272426

/-- The initial price of Sandy's scooter -/
def initial_price : ℝ := 800

/-- The cost of repairs -/
def repair_cost : ℝ := 200

/-- The selling price of the scooter -/
def selling_price : ℝ := 1200

/-- The gain percentage -/
def gain_percent : ℝ := 20

theorem sandy_scooter_price :
  ∃ (P : ℝ),
    P = initial_price ∧
    selling_price = (1 + gain_percent / 100) * (P + repair_cost) :=
by sorry

end sandy_scooter_price_l2724_272426


namespace tabitha_honey_per_cup_l2724_272482

/-- Proves that Tabitha adds 1 serving of honey per cup of tea -/
theorem tabitha_honey_per_cup :
  ∀ (cups_per_night : ℕ) 
    (container_ounces : ℕ) 
    (servings_per_ounce : ℕ) 
    (nights : ℕ),
  cups_per_night = 2 →
  container_ounces = 16 →
  servings_per_ounce = 6 →
  nights = 48 →
  (container_ounces * servings_per_ounce) / (cups_per_night * nights) = 1 :=
by
  sorry

#check tabitha_honey_per_cup

end tabitha_honey_per_cup_l2724_272482


namespace always_positive_l2724_272430

theorem always_positive (x y : ℝ) : 3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13 > 0 := by
  sorry

end always_positive_l2724_272430


namespace prob_irrational_not_adjacent_l2724_272457

/-- The number of rational terms in the expansion of (x + 2/√x)^6 -/
def num_rational_terms : ℕ := 4

/-- The number of irrational terms in the expansion of (x + 2/√x)^6 -/
def num_irrational_terms : ℕ := 3

/-- The total number of terms in the expansion -/
def total_terms : ℕ := num_rational_terms + num_irrational_terms

/-- The probability that irrational terms are not adjacent in the expansion of (x + 2/√x)^6 -/
theorem prob_irrational_not_adjacent : 
  (Nat.factorial num_rational_terms * (Nat.factorial (num_rational_terms + 1)) / 
   Nat.factorial num_irrational_terms) / 
  (Nat.factorial total_terms) = 2 / 7 := by
  sorry

end prob_irrational_not_adjacent_l2724_272457


namespace sum_reciprocals_bound_l2724_272447

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∀ M : ℝ, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 1 / a' + 1 / b' > M :=
by sorry

end sum_reciprocals_bound_l2724_272447


namespace other_coin_denomination_l2724_272406

/-- Given the following conditions:
    - There are 344 coins in total
    - The total value of all coins is 7100 paise (Rs. 71)
    - There are 300 coins of 20 paise each
    - There are two types of coins: 20 paise and another unknown denomination
    Prove that the denomination of the other type of coin is 25 paise -/
theorem other_coin_denomination
  (total_coins : ℕ)
  (total_value : ℕ)
  (twenty_paise_coins : ℕ)
  (h_total_coins : total_coins = 344)
  (h_total_value : total_value = 7100)
  (h_twenty_paise_coins : twenty_paise_coins = 300)
  : (total_value - twenty_paise_coins * 20) / (total_coins - twenty_paise_coins) = 25 := by
  sorry

end other_coin_denomination_l2724_272406


namespace polynomial_divisibility_l2724_272412

def p (x : ℝ) : ℝ := 4 * x^3 - 8 * x^2 + 8 * x - 16

theorem polynomial_divisibility :
  (∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * q x) ∧
  (∃ r : ℝ → ℝ, ∀ x, p x = (x^2 + 1) * r x) :=
sorry

end polynomial_divisibility_l2724_272412


namespace factory_price_decrease_and_sales_optimization_l2724_272498

/-- The average decrease rate in factory price over two years -/
def average_decrease_rate : ℝ := 0.1

/-- The price reduction that maximizes sales while maintaining the target profit -/
def price_reduction : ℝ := 15

/-- The initial factory price in 2019 -/
def initial_price : ℝ := 200

/-- The final factory price in 2021 -/
def final_price : ℝ := 162

/-- The initial number of pieces sold per day at the original price -/
def initial_sales : ℝ := 20

/-- The increase in sales for every 5 yuan reduction in price -/
def sales_increase_rate : ℝ := 2

/-- The target daily profit after price reduction -/
def target_profit : ℝ := 1150

theorem factory_price_decrease_and_sales_optimization :
  (initial_price * (1 - average_decrease_rate)^2 = final_price) ∧
  ((initial_price - final_price - price_reduction) * 
   (initial_sales + sales_increase_rate * price_reduction) = target_profit) ∧
  (∀ m : ℝ, m > price_reduction → 
   ((initial_price - final_price - m) * (initial_sales + sales_increase_rate * m) ≠ target_profit)) :=
by sorry

end factory_price_decrease_and_sales_optimization_l2724_272498


namespace abs_plus_a_neg_implies_b_sq_lt_a_sq_but_not_conversely_l2724_272479

theorem abs_plus_a_neg_implies_b_sq_lt_a_sq_but_not_conversely :
  ∃ (a b : ℝ), (abs b + a < 0 → b^2 < a^2) ∧
  ¬(∀ (a b : ℝ), b^2 < a^2 → abs b + a < 0) :=
by sorry

end abs_plus_a_neg_implies_b_sq_lt_a_sq_but_not_conversely_l2724_272479


namespace highlighter_count_after_increase_l2724_272422

/-- Calculates the final number of highlighters after accounting for broken and borrowed ones, 
    and applying a 25% increase. -/
theorem highlighter_count_after_increase 
  (pink yellow blue green purple : ℕ)
  (broken_pink broken_yellow broken_blue : ℕ)
  (borrowed_green borrowed_purple : ℕ)
  (h1 : pink = 18)
  (h2 : yellow = 14)
  (h3 : blue = 11)
  (h4 : green = 8)
  (h5 : purple = 7)
  (h6 : broken_pink = 3)
  (h7 : broken_yellow = 2)
  (h8 : broken_blue = 1)
  (h9 : borrowed_green = 1)
  (h10 : borrowed_purple = 2) :
  let remaining := (pink - broken_pink) + (yellow - broken_yellow) + (blue - broken_blue) +
                   (green - borrowed_green) + (purple - borrowed_purple)
  let increase := (remaining * 25) / 100
  (remaining + increase) = 61 :=
by sorry

end highlighter_count_after_increase_l2724_272422
