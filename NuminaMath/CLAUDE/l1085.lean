import Mathlib

namespace NUMINAMATH_CALUDE_regression_equation_proof_l1085_108561

/-- Given an exponential model and a regression line equation, 
    prove the resulting regression equation. -/
theorem regression_equation_proof 
  (y : ℝ → ℝ) 
  (k a : ℝ) 
  (h1 : ∀ x, y x = Real.exp (k * x + a)) 
  (h2 : ∀ x, 0.25 * x - 2.58 = Real.log (y x)) : 
  ∀ x, y x = Real.exp (0.25 * x - 2.58) := by
sorry

end NUMINAMATH_CALUDE_regression_equation_proof_l1085_108561


namespace NUMINAMATH_CALUDE_sum_of_integers_l1085_108501

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 4) (h2 : a * b = 63) : a + b = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1085_108501


namespace NUMINAMATH_CALUDE_system_solution_l1085_108588

theorem system_solution (x y z : ℝ) : 
  ((x + 1) * y * z = 12 ∧ 
   (y + 1) * z * x = 4 ∧ 
   (z + 1) * x * y = 4) ↔ 
  ((x = 2 ∧ y = -2 ∧ z = -2) ∨ 
   (x = 1/3 ∧ y = 3 ∧ z = 3)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1085_108588


namespace NUMINAMATH_CALUDE_watchman_max_demand_l1085_108535

/-- The amount of the bet made by the trespasser with his friends -/
def bet_amount : ℕ := 100

/-- The trespasser's net loss if he pays the watchman -/
def net_loss_if_pay (amount : ℕ) : ℤ := amount - bet_amount

/-- The trespasser's net loss if he doesn't pay the watchman -/
def net_loss_if_not_pay : ℕ := bet_amount

/-- Predicate to determine if the trespasser will pay for a given amount -/
def will_pay (amount : ℕ) : Prop :=
  net_loss_if_pay amount < net_loss_if_not_pay

/-- The maximum amount the watchman can demand -/
def max_demand : ℕ := 199

theorem watchman_max_demand :
  (∀ n : ℕ, n ≤ max_demand → will_pay n) ∧
  (∀ n : ℕ, n > max_demand → ¬will_pay n) :=
sorry

end NUMINAMATH_CALUDE_watchman_max_demand_l1085_108535


namespace NUMINAMATH_CALUDE_acrobats_count_correct_l1085_108520

/-- The number of acrobats at the farm. -/
def num_acrobats : ℕ := 13

/-- The number of elephants at the farm. -/
def num_elephants : ℕ := sorry

/-- The number of horses at the farm. -/
def num_horses : ℕ := sorry

/-- The total number of legs at the farm. -/
def total_legs : ℕ := 54

/-- The total number of heads at the farm. -/
def total_heads : ℕ := 20

/-- Theorem stating that the number of acrobats is correct given the conditions. -/
theorem acrobats_count_correct :
  2 * num_acrobats + 4 * num_elephants + 4 * num_horses = total_legs ∧
  num_acrobats + num_elephants + num_horses = total_heads ∧
  num_acrobats = 13 := by
  sorry


end NUMINAMATH_CALUDE_acrobats_count_correct_l1085_108520


namespace NUMINAMATH_CALUDE_sues_necklace_beads_l1085_108555

theorem sues_necklace_beads (purple : ℕ) (blue : ℕ) (green : ℕ) 
  (h1 : purple = 7)
  (h2 : blue = 2 * purple)
  (h3 : green = blue + 11) :
  purple + blue + green = 46 := by
  sorry

end NUMINAMATH_CALUDE_sues_necklace_beads_l1085_108555


namespace NUMINAMATH_CALUDE_triangle_area_l1085_108557

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 1 ∧ 
  t.b = Real.sqrt 3 ∧ 
  t.A + t.C = 2 * t.B

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h : triangle_conditions t) : 
  (1/2 : Real) * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l1085_108557


namespace NUMINAMATH_CALUDE_abc_inequality_l1085_108536

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c) (sum_sq : a^2 + b^2 + c^2 = 9) :
  a * b * c + 1 > 3 * a :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l1085_108536


namespace NUMINAMATH_CALUDE_find_a_value_l1085_108594

/-- Given sets A and B, prove that a is either -2/3 or -7/4 -/
theorem find_a_value (x : ℝ) (a : ℝ) : 
  let A : Set ℝ := {1, 2, x^2 - 5*x + 9}
  let B : Set ℝ := {3, x^2 + a*x + a}
  A = {1, 2, 3} → 2 ∈ B → (a = -2/3 ∨ a = -7/4) := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l1085_108594


namespace NUMINAMATH_CALUDE_min_value_rational_function_l1085_108546

theorem min_value_rational_function :
  (∀ x : ℝ, x > -1 → ((x^2 + 7*x + 10) / (x + 1)) ≥ 9) ∧
  (∃ x : ℝ, x > -1 ∧ ((x^2 + 7*x + 10) / (x + 1)) = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_rational_function_l1085_108546


namespace NUMINAMATH_CALUDE_inequality_solution_l1085_108567

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - (a + 3) * x + 2 * (a + 1) ≥ 0 ↔ 
    (a ≥ 1 ∧ (x ≥ a + 1 ∨ x ≤ 2)) ∨ 
    (a < 1 ∧ (x ≥ 2 ∨ x ≤ a + 1))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1085_108567


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l1085_108571

/-- The polynomial p(x) = ax^4 + bx^3 + 20x^2 - 12x + 10 -/
def p (a b : ℚ) (x : ℚ) : ℚ := a * x^4 + b * x^3 + 20 * x^2 - 12 * x + 10

/-- The factor q(x) = 2x^2 + 3x - 4 -/
def q (x : ℚ) : ℚ := 2 * x^2 + 3 * x - 4

/-- Theorem stating that if q(x) is a factor of p(x), then a = 2 and b = 27 -/
theorem polynomial_factor_implies_coefficients (a b : ℚ) :
  (∃ r : ℚ → ℚ, ∀ x, p a b x = q x * r x) → a = 2 ∧ b = 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l1085_108571


namespace NUMINAMATH_CALUDE_problem_solution_l1085_108560

theorem problem_solution (x : ℝ) (h : 3 * x^2 - x = 1) :
  6 * x^3 + 7 * x^2 - 5 * x + 2010 = 2013 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1085_108560


namespace NUMINAMATH_CALUDE_distinct_collections_count_l1085_108529

/-- Represents the count of each letter in "MATHEMATICAL" --/
structure LetterCount where
  a : Nat
  e : Nat
  i : Nat
  t : Nat
  m : Nat
  h : Nat
  l : Nat
  c : Nat

/-- The initial count of letters in "MATHEMATICAL" --/
def initialCount : LetterCount := {
  a := 3, e := 1, i := 1,
  t := 2, m := 2, h := 1, l := 1, c := 2
}

/-- A collection of letters that fell off --/
structure FallenLetters where
  vowels : Finset Char
  consonants : Finset Char

/-- Checks if a collection of fallen letters is valid --/
def isValidCollection (letters : FallenLetters) : Prop :=
  letters.vowels.card = 3 ∧ letters.consonants.card = 3

/-- Counts distinct collections considering indistinguishable letters --/
def countDistinctCollections (count : LetterCount) : Nat :=
  sorry

theorem distinct_collections_count :
  countDistinctCollections initialCount = 80 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l1085_108529


namespace NUMINAMATH_CALUDE_pasture_perimeter_difference_l1085_108513

/-- Calculates the perimeter of a pasture given the number of stakes and the interval between stakes -/
def pasture_perimeter (stakes : ℕ) (interval : ℕ) : ℕ := stakes * interval

/-- The difference between the perimeters of two pastures -/
theorem pasture_perimeter_difference : 
  pasture_perimeter 82 20 - pasture_perimeter 96 10 = 680 := by
  sorry

end NUMINAMATH_CALUDE_pasture_perimeter_difference_l1085_108513


namespace NUMINAMATH_CALUDE_fraction_problem_l1085_108506

/-- The fraction of p's amount that q and r each have -/
def fraction_of_p (p q r : ℚ) : ℚ :=
  q / p

/-- The problem statement -/
theorem fraction_problem (p q r : ℚ) : 
  p = 56 → 
  p = 2 * (fraction_of_p p q r) * p + 42 → 
  q = r → 
  fraction_of_p p q r = 1/8 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1085_108506


namespace NUMINAMATH_CALUDE_sine_cosine_cube_difference_l1085_108532

theorem sine_cosine_cube_difference (α : ℝ) (n : ℝ) 
  (h : Real.sin α - Real.cos α = n) : 
  Real.sin α ^ 3 - Real.cos α ^ 3 = (3 * n - n^3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_cube_difference_l1085_108532


namespace NUMINAMATH_CALUDE_egg_purchase_cost_l1085_108538

def dozen : ℕ := 12
def egg_price : ℚ := 0.50

theorem egg_purchase_cost (num_dozens : ℕ) : 
  (num_dozens * dozen * egg_price : ℚ) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_egg_purchase_cost_l1085_108538


namespace NUMINAMATH_CALUDE_picture_placement_l1085_108580

/-- Given a wall of width 30 feet, with two pictures each 4 feet wide and spaced 1 foot apart
    hung in the center, the distance from the end of the wall to the nearest edge of the first
    picture is 10.5 feet. -/
theorem picture_placement (wall_width : ℝ) (picture_width : ℝ) (picture_space : ℝ)
  (h_wall : wall_width = 30)
  (h_picture : picture_width = 4)
  (h_space : picture_space = 1) :
  let total_picture_space := 2 * picture_width + picture_space
  (wall_width - total_picture_space) / 2 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_picture_placement_l1085_108580


namespace NUMINAMATH_CALUDE_card_drawing_probability_ratio_l1085_108503

theorem card_drawing_probability_ratio :
  let total_cards : ℕ := 60
  let num_range : ℕ := 15
  let cards_per_num : ℕ := 4
  let draw_count : ℕ := 4

  let p : ℚ := (num_range : ℚ) / (Nat.choose total_cards draw_count)
  let q : ℚ := (num_range * (num_range - 1) * Nat.choose cards_per_num 3 * Nat.choose cards_per_num 1 : ℚ) / 
                (Nat.choose total_cards draw_count)

  q / p = 224 := by sorry

end NUMINAMATH_CALUDE_card_drawing_probability_ratio_l1085_108503


namespace NUMINAMATH_CALUDE_quadratic_solution_set_theorem_l1085_108507

/-- Given a quadratic function f(x) = ax² + bx + c, 
    this is the type of its solution set when f(x) > 0 -/
def QuadraticSolutionSet (a b c : ℝ) := Set ℝ

/-- The condition that the solution set of ax² + bx + c > 0 
    is the open interval (3, 6) -/
def SolutionSetCondition (a b c : ℝ) : Prop :=
  QuadraticSolutionSet a b c = {x : ℝ | 3 < x ∧ x < 6}

theorem quadratic_solution_set_theorem 
  (a b c : ℝ) (h : SolutionSetCondition a b c) :
  QuadraticSolutionSet c b a = {x : ℝ | x < 1/6 ∨ x > 1/3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_theorem_l1085_108507


namespace NUMINAMATH_CALUDE_min_value_of_function_l1085_108559

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (x^2 + x + 25) / x ≥ 11 ∧ ∃ y > 0, (y^2 + y + 25) / y = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1085_108559


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1085_108562

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 1)) ↔ x ≠ -1 :=
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1085_108562


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1085_108530

/-- Given two lines l₁ and l₂, prove that if they are perpendicular, then a = 0 or a = 5/3 -/
theorem perpendicular_lines_a_values (a : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | a * x + 3 * y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | 2 * x + (a^2 - a) * y + 3 = 0}
  let perpendicular := ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁, y₁) ∈ l₁ → (x₂, y₂) ∈ l₂ → 
    (x₂ - x₁) * (a * (x₂ - x₁) + 3 * (y₂ - y₁)) + 
    (y₂ - y₁) * (2 * (x₂ - x₁) + (a^2 - a) * (y₂ - y₁)) = 0
  perpendicular → a = 0 ∨ a = 5/3 :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l1085_108530


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1085_108525

/-- Geometric sequence with first term a and common ratio r -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r^(n - 1)

/-- Sequence b_n as defined in the problem -/
def b_sequence (a : ℕ → ℝ) : ℕ → ℝ := 
  fun n => (Finset.range n).sum (fun k => (n - k) * a (k + 1))

/-- Sum of first n terms of a sequence -/
def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range n).sum (fun k => a (k + 1))

theorem geometric_sequence_problem (m : ℝ) (h_m : m ≠ 0) :
  ∃ (a : ℕ → ℝ), 
    (∃ r, a = geometric_sequence m r) ∧ 
    b_sequence a 1 = m ∧
    b_sequence a 2 = 3/2 * m ∧
    (∀ n : ℕ, n > 0 → 1 ≤ sequence_sum a n ∧ sequence_sum a n ≤ 3) →
    (∀ n, a n = m * (-1/2)^(n-1)) ∧
    (2 ≤ m ∧ m ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1085_108525


namespace NUMINAMATH_CALUDE_sum_cube_inequality_l1085_108573

theorem sum_cube_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_cube_inequality_l1085_108573


namespace NUMINAMATH_CALUDE_paper_strips_length_l1085_108515

/-- The total length of overlapping paper strips -/
def total_length (n : ℕ) (sheet_length : ℝ) (overlap : ℝ) : ℝ :=
  sheet_length + (n - 1) * (sheet_length - overlap)

/-- Theorem: The total length of 30 sheets of 25 cm paper strips overlapped by 6 cm is 576 cm -/
theorem paper_strips_length :
  total_length 30 25 6 = 576 := by
  sorry

end NUMINAMATH_CALUDE_paper_strips_length_l1085_108515


namespace NUMINAMATH_CALUDE_string_cutting_l1085_108565

/-- Proves that cutting off 1/4 of a 2/3 meter long string leaves 50 cm remaining. -/
theorem string_cutting (string_length : ℚ) (h1 : string_length = 2/3) :
  (string_length * 100 - (1/4 * string_length * 100)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_string_cutting_l1085_108565


namespace NUMINAMATH_CALUDE_only_three_four_five_is_right_triangle_l1085_108593

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem only_three_four_five_is_right_triangle :
  (¬ is_right_triangle 1 2 3) ∧
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  (¬ is_right_triangle 1 2 3) :=
sorry

end NUMINAMATH_CALUDE_only_three_four_five_is_right_triangle_l1085_108593


namespace NUMINAMATH_CALUDE_modulus_of_z_l1085_108537

theorem modulus_of_z (z : ℂ) (h : (z + Complex.I) * Complex.I = -3 + 4 * Complex.I) : 
  Complex.abs z = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1085_108537


namespace NUMINAMATH_CALUDE_max_value_expression_l1085_108599

theorem max_value_expression (x y : ℝ) : 
  (2 * x + Real.sqrt 2 * y) / (2 * x^4 + 4 * y^4 + 9) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1085_108599


namespace NUMINAMATH_CALUDE_multiply_negative_four_with_three_halves_l1085_108576

theorem multiply_negative_four_with_three_halves : (-4 : ℚ) * (3/2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_negative_four_with_three_halves_l1085_108576


namespace NUMINAMATH_CALUDE_shortest_distance_on_cube_face_l1085_108583

/-- The shortest distance on the surface of a cube between midpoints of opposite edges on the same face -/
theorem shortest_distance_on_cube_face (edge_length : ℝ) (h : edge_length = 2) :
  let midpoint_distance := Real.sqrt 2
  ∃ (path : ℝ), path ≥ midpoint_distance ∧
    (∀ (other_path : ℝ), other_path ≥ midpoint_distance → path ≤ other_path) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_on_cube_face_l1085_108583


namespace NUMINAMATH_CALUDE_class_size_is_30_l1085_108519

/-- Represents the capacity of a hotel room -/
structure RoomCapacity where
  queen_bed_capacity : Nat
  queen_beds : Nat
  couch_capacity : Nat

/-- Calculates the total number of students in a class given room requirements -/
def calculate_class_size (room_capacity : RoomCapacity) (rooms_booked : Nat) : Nat :=
  (room_capacity.queen_bed_capacity * room_capacity.queen_beds + room_capacity.couch_capacity) * rooms_booked

/-- Theorem stating that the class size is 30 given the specific room configuration and booking requirements -/
theorem class_size_is_30 :
  let room_capacity : RoomCapacity := { queen_bed_capacity := 2, queen_beds := 2, couch_capacity := 1 }
  let rooms_booked := 6
  calculate_class_size room_capacity rooms_booked = 30 := by
  sorry


end NUMINAMATH_CALUDE_class_size_is_30_l1085_108519


namespace NUMINAMATH_CALUDE_point_on_line_l1085_108590

/-- If (m, n) and (m + 2, n + k) are two points on the line with equation x = 2y + 3, then k = 1 -/
theorem point_on_line (m n k : ℝ) : 
  (m = 2*n + 3) → 
  (m + 2 = 2*(n + k) + 3) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1085_108590


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1085_108526

theorem fraction_subtraction : 
  (3 + 7 + 11) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 7 + 11) = 33 / 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1085_108526


namespace NUMINAMATH_CALUDE_amusement_park_spending_l1085_108595

/-- Calculates the total amount spent by a group of children at an amusement park -/
def total_spent (num_children : ℕ) 
  (ferris_wheel_cost ferris_wheel_riders : ℕ)
  (roller_coaster_cost roller_coaster_riders : ℕ)
  (merry_go_round_cost : ℕ)
  (bumper_cars_cost bumper_cars_riders : ℕ)
  (ice_cream_cost ice_cream_eaters : ℕ)
  (hot_dog_cost hot_dog_eaters : ℕ)
  (pizza_cost pizza_eaters : ℕ) : ℕ :=
  ferris_wheel_cost * ferris_wheel_riders +
  roller_coaster_cost * roller_coaster_riders +
  merry_go_round_cost * num_children +
  bumper_cars_cost * bumper_cars_riders +
  ice_cream_cost * ice_cream_eaters +
  hot_dog_cost * hot_dog_eaters +
  pizza_cost * pizza_eaters

/-- Theorem stating that the total amount spent by the group is $170 -/
theorem amusement_park_spending :
  total_spent 8 5 5 7 3 3 4 6 8 5 6 4 4 3 = 170 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_spending_l1085_108595


namespace NUMINAMATH_CALUDE_triangle_equality_l1085_108504

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  h : A + B + C = 180  -- Sum of angles in a triangle is 180°

-- Define the theorem
theorem triangle_equality (t : Triangle) 
  (h₁ : t.A > t.B)  -- A > B
  (h₂ : ∃ (C₁ C₂ : ℝ), C₁ + C₂ = t.C ∧ C₁ = 2 * C₂)  -- C₁ + C₂ = C and C₁ = 2C₂
  : t.A = t.B := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l1085_108504


namespace NUMINAMATH_CALUDE_total_votes_l1085_108508

/-- Proves that the total number of votes is 290 given the specified conditions -/
theorem total_votes (votes_against : ℕ) (votes_in_favor : ℕ) (total_votes : ℕ) : 
  votes_in_favor = votes_against + 58 →
  votes_against = (40 * total_votes) / 100 →
  total_votes = votes_in_favor + votes_against →
  total_votes = 290 := by
sorry

end NUMINAMATH_CALUDE_total_votes_l1085_108508


namespace NUMINAMATH_CALUDE_vlad_sister_height_difference_l1085_108553

/-- The height difference between two people given their heights in centimeters -/
def height_difference (height1 : ℝ) (height2 : ℝ) : ℝ :=
  height1 - height2

/-- Vlad's height in centimeters -/
def vlad_height : ℝ := 190.5

/-- Vlad's sister's height in centimeters -/
def sister_height : ℝ := 86.36

/-- Theorem: The height difference between Vlad and his sister is 104.14 centimeters -/
theorem vlad_sister_height_difference :
  height_difference vlad_height sister_height = 104.14 := by
  sorry


end NUMINAMATH_CALUDE_vlad_sister_height_difference_l1085_108553


namespace NUMINAMATH_CALUDE_parabola_intersection_area_l1085_108550

/-- Parabola represented by y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Line passing through the focus at a 45° angle -/
def Line (x y : ℝ) : Prop := y = x - 1

/-- Intersection points of the line with the parabola -/
def IntersectionPoints (A B : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ Line A.1 A.2 ∧ Line B.1 B.2

/-- Origin point -/
def Origin : ℝ × ℝ := (0, 0)

/-- Area of a triangle given three points -/
def TriangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem parabola_intersection_area :
  ∀ A B : ℝ × ℝ, IntersectionPoints A B →
  TriangleArea Origin A B = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_area_l1085_108550


namespace NUMINAMATH_CALUDE_square_side_length_l1085_108541

theorem square_side_length (s : ℝ) : s > 0 → s^2 = 3 * (4 * s) → s = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1085_108541


namespace NUMINAMATH_CALUDE_locus_of_centers_l1085_108597

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ with equation (x - 3)² + y² = 9 -/
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

/-- A circle is externally tangent to C₁ if the distance between their centers
    equals the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers
    equals the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (3 - r)^2

/-- The locus of centers (a,b) of circles externally tangent to C₁ and internally tangent to C₂
    satisfies the equation 28a² + 64b² - 84a - 49 = 0 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) → 
  28 * a^2 + 64 * b^2 - 84 * a - 49 = 0 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_centers_l1085_108597


namespace NUMINAMATH_CALUDE_cambridge_population_l1085_108512

theorem cambridge_population : ∃ (p : ℕ), p > 0 ∧ (
  ∀ (w a : ℚ),
  w > 0 ∧ a > 0 ∧
  w + a = 12 * p ∧
  w / 6 + a / 8 = 12 →
  p = 7
) := by
  sorry

end NUMINAMATH_CALUDE_cambridge_population_l1085_108512


namespace NUMINAMATH_CALUDE_work_completion_time_l1085_108548

theorem work_completion_time (x : ℝ) : 
  x > 0 →  -- p's completion time is positive
  (2 / x + 3 * (1 / x + 1 / 6) = 1) →  -- work equation
  x = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1085_108548


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1085_108534

/-- Given a geometric sequence with common ratio 2 and sum of first four terms equal to 1,
    the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a : ℝ) : 
  (∃ (S₄ S₈ : ℝ), 
    S₄ = a * (1 + 2 + 2^2 + 2^3) ∧
    S₄ = 1 ∧
    S₈ = a * (1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7)) →
  (∃ S₈ : ℝ, S₈ = a * (1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7) ∧ S₈ = 17) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1085_108534


namespace NUMINAMATH_CALUDE_complex_modulus_l1085_108547

theorem complex_modulus (x y : ℝ) (h : (1 + Complex.I) * x = 1 - Complex.I * y) :
  Complex.abs (x - Complex.I * y) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1085_108547


namespace NUMINAMATH_CALUDE_marble_ratio_l1085_108514

/-- Represents the number of marbles each person has -/
structure Marbles where
  you : ℕ
  brother : ℕ
  friend : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : Marbles) : Prop :=
  m.you = 16 ∧
  m.you + m.brother + m.friend = 63 ∧
  m.you - 2 = 2 * (m.brother + 2) ∧
  ∃ k : ℕ, m.friend = k * m.you

/-- The theorem to prove -/
theorem marble_ratio (m : Marbles) (h : marble_problem m) :
  m.friend * 8 = m.you * 21 := by
  sorry


end NUMINAMATH_CALUDE_marble_ratio_l1085_108514


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1085_108564

/-- Calculates the area of the shaded region in a grid with two unshaded triangles. -/
theorem shaded_area_calculation (grid_width grid_height : ℝ)
  (large_triangle_base large_triangle_height : ℝ)
  (small_triangle_base small_triangle_height : ℝ)
  (h1 : grid_width = 15)
  (h2 : grid_height = 5)
  (h3 : large_triangle_base = grid_width)
  (h4 : large_triangle_height = grid_height)
  (h5 : small_triangle_base = 3)
  (h6 : small_triangle_height = 2) :
  grid_width * grid_height - (1/2 * large_triangle_base * large_triangle_height) -
  (1/2 * small_triangle_base * small_triangle_height) = 34.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1085_108564


namespace NUMINAMATH_CALUDE_certain_number_exists_and_unique_l1085_108587

theorem certain_number_exists_and_unique : 
  ∃! x : ℝ, x / 5 + x + 5 = 65 := by sorry

end NUMINAMATH_CALUDE_certain_number_exists_and_unique_l1085_108587


namespace NUMINAMATH_CALUDE_boat_upstream_speed_l1085_108502

/-- Proves that the upstream speed is approximately 29.82 miles per hour given the conditions of the boat problem -/
theorem boat_upstream_speed (distance : ℝ) (downstream_time : ℝ) (time_difference : ℝ) :
  distance = 90 ∧ 
  downstream_time = 2.5191640969412834 ∧ 
  time_difference = 0.5 →
  ∃ upstream_speed : ℝ, 
    distance = upstream_speed * (downstream_time + time_difference) ∧
    abs (upstream_speed - 29.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_speed_l1085_108502


namespace NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l1085_108589

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x^2 = 1 is a quadratic equation in one variable -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l1085_108589


namespace NUMINAMATH_CALUDE_reciprocal_of_fraction_difference_l1085_108581

theorem reciprocal_of_fraction_difference : (((2 : ℚ) / 3 - (3 : ℚ) / 4)⁻¹ : ℚ) = -12 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_fraction_difference_l1085_108581


namespace NUMINAMATH_CALUDE_y_range_l1085_108575

theorem y_range (y : ℝ) (h1 : y > 0) (h2 : Real.log y / Real.log 3 ≤ 3 - Real.log (9 * y) / Real.log 3) : 
  0 < y ∧ y ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_range_l1085_108575


namespace NUMINAMATH_CALUDE_negative_three_a_plus_two_a_equals_negative_a_l1085_108591

theorem negative_three_a_plus_two_a_equals_negative_a (a : ℝ) : -3*a + 2*a = -a := by
  sorry

end NUMINAMATH_CALUDE_negative_three_a_plus_two_a_equals_negative_a_l1085_108591


namespace NUMINAMATH_CALUDE_B_power_101_l1085_108505

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  !![1, 0, 0;
     0, 0, 1;
     0, 1, 0]

theorem B_power_101 : B^101 = B := by sorry

end NUMINAMATH_CALUDE_B_power_101_l1085_108505


namespace NUMINAMATH_CALUDE_train_length_l1085_108569

/-- Given a train that crosses a 300-meter platform in 36 seconds and a signal pole in 18 seconds, 
    prove that the length of the train is 300 meters. -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
    (h1 : platform_length = 300)
    (h2 : platform_time = 36)
    (h3 : pole_time = 18) : 
  let train_length := (platform_length * pole_time) / (platform_time - pole_time)
  train_length = 300 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1085_108569


namespace NUMINAMATH_CALUDE_line_equation_and_sum_l1085_108531

/-- Given a line with slope 5 passing through (-2, 4), prove its equation and m + b value -/
theorem line_equation_and_sum (m b : ℝ) : 
  m = 5 → -- Given slope
  4 = m * (-2) + b → -- Point (-2, 4) lies on the line
  (∀ x y, y = m * x + b ↔ y - 4 = m * (x + 2)) → -- Line equation
  m + b = 19 := by
sorry

end NUMINAMATH_CALUDE_line_equation_and_sum_l1085_108531


namespace NUMINAMATH_CALUDE_orange_harvest_problem_l1085_108586

/-- Proves that the number of sacks harvested per day is 66, given the conditions of the orange harvest problem. -/
theorem orange_harvest_problem (oranges_per_sack : ℕ) (harvest_days : ℕ) (total_oranges : ℕ) 
  (h1 : oranges_per_sack = 25)
  (h2 : harvest_days = 87)
  (h3 : total_oranges = 143550) :
  total_oranges / (oranges_per_sack * harvest_days) = 66 := by
  sorry

#eval 143550 / (25 * 87)  -- Should output 66

end NUMINAMATH_CALUDE_orange_harvest_problem_l1085_108586


namespace NUMINAMATH_CALUDE_negative_expressions_l1085_108554

/-- Represents a number with an approximate value -/
structure ApproxNumber where
  value : ℝ

/-- Given approximate values for P, Q, R, S, and T -/
def P : ApproxNumber := ⟨-4.2⟩
def Q : ApproxNumber := ⟨-2.3⟩
def R : ApproxNumber := ⟨0⟩
def S : ApproxNumber := ⟨1.1⟩
def T : ApproxNumber := ⟨2.7⟩

/-- Helper function to extract the value from ApproxNumber -/
def getValue (x : ApproxNumber) : ℝ := x.value

/-- Theorem stating which expressions are negative -/
theorem negative_expressions :
  (getValue P - getValue Q < 0) ∧
  (getValue P + getValue T < 0) ∧
  (getValue P * getValue Q ≥ 0) ∧
  ((getValue S / getValue Q) * getValue P ≥ 0) ∧
  (getValue R / (getValue P * getValue Q) ≥ 0) ∧
  ((getValue S + getValue T) / getValue R ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_negative_expressions_l1085_108554


namespace NUMINAMATH_CALUDE_tailor_buttons_count_l1085_108558

/-- The number of green buttons purchased by the tailor -/
def green_buttons : ℕ := 90

/-- The number of yellow buttons purchased by the tailor -/
def yellow_buttons : ℕ := green_buttons + 10

/-- The number of blue buttons purchased by the tailor -/
def blue_buttons : ℕ := green_buttons - 5

/-- The total number of buttons purchased by the tailor -/
def total_buttons : ℕ := green_buttons + yellow_buttons + blue_buttons

theorem tailor_buttons_count : total_buttons = 275 := by
  sorry

end NUMINAMATH_CALUDE_tailor_buttons_count_l1085_108558


namespace NUMINAMATH_CALUDE_solution_sets_intersection_and_union_l1085_108566

def equation1 (p : ℝ) (x : ℝ) : Prop := x^2 - p*x + 6 = 0

def equation2 (q : ℝ) (x : ℝ) : Prop := x^2 + 6*x - q = 0

def solution_set (equation : ℝ → Prop) : Set ℝ :=
  {x | equation x}

theorem solution_sets_intersection_and_union
  (p q : ℝ)
  (M : Set ℝ)
  (N : Set ℝ)
  (h1 : M = solution_set (equation1 p))
  (h2 : N = solution_set (equation2 q))
  (h3 : M ∩ N = {2}) :
  p = 5 ∧ q = 16 ∧ M ∪ N = {2, 3, -8} := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_intersection_and_union_l1085_108566


namespace NUMINAMATH_CALUDE_money_left_l1085_108570

def salary_distribution (S : ℝ) : Prop :=
  let house_rent := (2/5) * S
  let food := (3/10) * S
  let conveyance := (1/8) * S
  let food_and_conveyance := food + conveyance
  food_and_conveyance = 3399.999999999999

theorem money_left (S : ℝ) (h : salary_distribution S) : 
  S - ((2/5 + 3/10 + 1/8) * S) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l1085_108570


namespace NUMINAMATH_CALUDE_divisibility_of_P_and_Q_l1085_108544

/-- Given that there exists a natural number n such that 1997 divides 111...1 (n ones),
    prove that 1997 divides both P and Q. -/
theorem divisibility_of_P_and_Q (n : ℕ) (h : ∃ k : ℕ, (10^n - 1) / 9 = 1997 * k) :
  ∃ (p q : ℕ), P = 1997 * p ∧ Q = 1997 * q :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_P_and_Q_l1085_108544


namespace NUMINAMATH_CALUDE_marys_birthday_money_l1085_108539

theorem marys_birthday_money (M : ℚ) : 
  (3/4 : ℚ) * M - (1/5 : ℚ) * ((3/4 : ℚ) * M) = 60 → M = 100 := by
  sorry

end NUMINAMATH_CALUDE_marys_birthday_money_l1085_108539


namespace NUMINAMATH_CALUDE_circle_center_range_l1085_108549

theorem circle_center_range (a : ℝ) : 
  let C : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + (p.2 - (a-2))^2 = 9}
  let M : ℝ × ℝ := (0, 3)
  (3, -2) ∈ C ∧ (0, -5) ∈ C ∧ 
  (∃ N ∈ C, (N.1 - M.1)^2 + (N.2 - M.2)^2 = 4 * ((N.1 - a)^2 + (N.2 - (a-2))^2)) →
  (-3 ≤ a ∧ a ≤ 0) ∨ (1 ≤ a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_circle_center_range_l1085_108549


namespace NUMINAMATH_CALUDE_probability_two_forks_one_spoon_one_knife_l1085_108552

/-- The number of forks in the drawer -/
def num_forks : ℕ := 8

/-- The number of spoons in the drawer -/
def num_spoons : ℕ := 5

/-- The number of knives in the drawer -/
def num_knives : ℕ := 7

/-- The total number of pieces of silverware -/
def total_silverware : ℕ := num_forks + num_spoons + num_knives

/-- The number of pieces to be drawn -/
def num_drawn : ℕ := 4

/-- The probability of drawing 2 forks, 1 spoon, and 1 knife -/
theorem probability_two_forks_one_spoon_one_knife :
  (Nat.choose num_forks 2 * Nat.choose num_spoons 1 * Nat.choose num_knives 1 : ℚ) /
  (Nat.choose total_silverware num_drawn : ℚ) = 196 / 969 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_forks_one_spoon_one_knife_l1085_108552


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1085_108528

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1085_108528


namespace NUMINAMATH_CALUDE_yoki_cans_count_l1085_108556

/-- Given a scenario where:
  - The total number of cans collected is 85
  - LaDonna picked up 25 cans
  - Prikya picked up twice as many cans as LaDonna
  - Yoki picked up the rest of the cans
This theorem proves that Yoki picked up 10 cans. -/
theorem yoki_cans_count (total : ℕ) (ladonna : ℕ) (prikya : ℕ) (yoki : ℕ) 
  (h1 : total = 85)
  (h2 : ladonna = 25)
  (h3 : prikya = 2 * ladonna)
  (h4 : total = ladonna + prikya + yoki) :
  yoki = 10 := by
  sorry

end NUMINAMATH_CALUDE_yoki_cans_count_l1085_108556


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1085_108540

-- Part 1
theorem problem_one : (2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3 := by
  sorry

-- Part 2
theorem problem_two : (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1/2) = -6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1085_108540


namespace NUMINAMATH_CALUDE_even_increasing_function_property_l1085_108572

/-- A function that is even on ℝ and increasing on (-∞, 0] -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

/-- Theorem stating that for an even function increasing on (-∞, 0],
    if f(a) ≤ f(2-a), then a ≥ 1 -/
theorem even_increasing_function_property (f : ℝ → ℝ) (a : ℝ) 
    (h1 : EvenIncreasingFunction f) (h2 : f a ≤ f (2 - a)) : 
    a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_function_property_l1085_108572


namespace NUMINAMATH_CALUDE_sum_of_geometric_sequences_indeterminate_l1085_108551

/-- A sequence is geometric if there exists a non-zero constant r such that each term is r times the previous term. -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = r * s n

/-- The sum of two sequences -/
def SequenceSum (s t : ℕ → ℝ) : ℕ → ℝ :=
  λ n => s n + t n

/-- Statement: Given two geometric sequences, their sum sequence may or may not be geometric or arithmetic. -/
theorem sum_of_geometric_sequences_indeterminate (a b : ℕ → ℝ)
    (ha : IsGeometricSequence a) (hb : IsGeometricSequence b) :
    ¬ (∀ a b : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence b →
      (IsGeometricSequence (SequenceSum a b) ∨
       ∃ d : ℝ, ∀ n : ℕ, SequenceSum a b (n + 1) = SequenceSum a b n + d)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_geometric_sequences_indeterminate_l1085_108551


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1085_108574

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := ℝ → ℝ

/-- The functional equation that f must satisfy -/
def SatisfiesEquation (f : PositiveRealFunction) (α : ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (f x + y) = α * x + 1 / f (1 / y)

theorem functional_equation_solution :
  ∀ α : ℝ, α ≠ 0 →
    (∃ f : PositiveRealFunction, SatisfiesEquation f α) ↔
    (α = 1 ∧ ∃ f : PositiveRealFunction, SatisfiesEquation f 1 ∧ ∀ x, x > 0 → f x = x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1085_108574


namespace NUMINAMATH_CALUDE_monthly_payment_difference_l1085_108516

/-- The cost of the house in dollars -/
def house_cost : ℕ := 480000

/-- The cost of the trailer in dollars -/
def trailer_cost : ℕ := 120000

/-- The loan term in years -/
def loan_term : ℕ := 20

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Calculates the monthly payment for a given cost over the loan term -/
def monthly_payment (cost : ℕ) : ℚ :=
  cost / (loan_term * months_per_year)

/-- The statement to be proved -/
theorem monthly_payment_difference :
  monthly_payment house_cost - monthly_payment trailer_cost = 1500 := by
  sorry

end NUMINAMATH_CALUDE_monthly_payment_difference_l1085_108516


namespace NUMINAMATH_CALUDE_smallest_angle_BFE_l1085_108518

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the incenter of a triangle
def incenter (t : Triangle) : Point := sorry

-- Define the measure of an angle
def angle_measure (p q r : Point) : ℝ := sorry

-- State the theorem
theorem smallest_angle_BFE (ABC : Triangle) :
  let D := incenter ABC
  let ABD := Triangle.mk ABC.A ABC.B D
  let E := incenter ABD
  let BDE := Triangle.mk ABC.B D E
  let F := incenter BDE
  ∃ (n : ℕ), 
    (∀ m : ℕ, m < n → ¬(∃ ABC : Triangle, angle_measure ABC.B F E = m)) ∧
    (∃ ABC : Triangle, angle_measure ABC.B F E = n) ∧
    n = 113 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_BFE_l1085_108518


namespace NUMINAMATH_CALUDE_not_all_projections_same_l1085_108542

/-- Represents a 3D shape -/
inductive Shape
  | Cube
  | Sphere
  | Cone

/-- Represents a type of orthographic projection -/
inductive Projection
  | FrontView
  | SideView
  | TopView

/-- Represents the result of an orthographic projection -/
inductive ProjectionResult
  | Square
  | Circle
  | IsoscelesTriangle

/-- Returns the projection result for a given shape and projection type -/
def projectShape (s : Shape) (p : Projection) : ProjectionResult :=
  match s, p with
  | Shape.Cube, _ => ProjectionResult.Square
  | Shape.Sphere, _ => ProjectionResult.Circle
  | Shape.Cone, Projection.TopView => ProjectionResult.Circle
  | Shape.Cone, _ => ProjectionResult.IsoscelesTriangle

/-- Theorem stating that it's not true that all projections are the same for all shapes -/
theorem not_all_projections_same : ¬ (∀ (s1 s2 : Shape) (p1 p2 : Projection), 
  projectShape s1 p1 = projectShape s2 p2) := by
  sorry


end NUMINAMATH_CALUDE_not_all_projections_same_l1085_108542


namespace NUMINAMATH_CALUDE_point_B_coordinates_l1085_108509

/-- Given point A (2, 4), vector a⃗ = (3, 4), and AB⃗ = 2a⃗, prove that the coordinates of point B are (8, 12). -/
theorem point_B_coordinates (A B : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (2, 4) →
  a = (3, 4) →
  B.1 - A.1 = 2 * a.1 →
  B.2 - A.2 = 2 * a.2 →
  B = (8, 12) := by
sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l1085_108509


namespace NUMINAMATH_CALUDE_journey_fraction_by_foot_l1085_108517

/-- Given a journey with specific conditions, proves the fraction traveled by foot -/
theorem journey_fraction_by_foot :
  let total_distance : ℝ := 30.000000000000007
  let bus_fraction : ℝ := 3/5
  let car_distance : ℝ := 2
  let foot_distance : ℝ := total_distance - bus_fraction * total_distance - car_distance
  foot_distance / total_distance = 1/3 := by
sorry

end NUMINAMATH_CALUDE_journey_fraction_by_foot_l1085_108517


namespace NUMINAMATH_CALUDE_equal_segments_exist_l1085_108543

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) :=
  (vertices : Fin (2*n) → ℝ × ℝ)
  (is_regular : sorry)

/-- A pairing of vertices in a regular polygon -/
def VertexPairing (n : ℕ) := Fin n → Fin (2*n) × Fin (2*n)

/-- The distance between two vertices in a regular polygon -/
def distance (p : RegularPolygon n) (i j : Fin (2*n)) : ℝ := sorry

theorem equal_segments_exist (m : ℕ) (n : ℕ) (h : n = 4*m + 2 ∨ n = 4*m + 3) 
  (p : RegularPolygon n) (pairing : VertexPairing n) : 
  ∃ (i j k l : Fin n), i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ j ≠ l ∧
    distance p (pairing i).1 (pairing i).2 = distance p (pairing k).1 (pairing k).2 :=
sorry

end NUMINAMATH_CALUDE_equal_segments_exist_l1085_108543


namespace NUMINAMATH_CALUDE_f_neg_two_eq_twelve_l1085_108577

/-- The polynomial function f(x) = x^5 + 4x^4 + x^2 + 20x + 16 -/
def f (x : ℝ) : ℝ := x^5 + 4*x^4 + x^2 + 20*x + 16

/-- Theorem: The value of f(-2) is 12 -/
theorem f_neg_two_eq_twelve : f (-2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_twelve_l1085_108577


namespace NUMINAMATH_CALUDE_modulus_of_z_l1085_108578

theorem modulus_of_z (z : ℂ) (h : z^2 = 48 - 14*I) : Complex.abs z = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1085_108578


namespace NUMINAMATH_CALUDE_PQ_length_l1085_108533

-- Define the triangles and their properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle : ℝ

-- Define the given triangles
def triangle_PQR : Triangle := {
  a := 8,   -- PR
  b := 10,  -- QR
  c := 5,   -- PQ (to be proved)
  angle := 60
}

def triangle_STU : Triangle := {
  a := 3,   -- SU
  b := 4,   -- TU (derived from similarity)
  c := 2,   -- ST
  angle := 60
}

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop :=
  t1.angle = t2.angle ∧ t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

-- Theorem statement
theorem PQ_length :
  similar triangle_PQR triangle_STU →
  triangle_PQR.c = 5 := by sorry

end NUMINAMATH_CALUDE_PQ_length_l1085_108533


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1085_108511

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1085_108511


namespace NUMINAMATH_CALUDE_unique_fibonacci_partition_l1085_108522

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def is_fibonacci (n : ℕ) : Prop := ∃ k, fibonacci k = n

def is_partition (A B : Set ℕ) : Prop :=
  (A ∩ B = ∅) ∧ (A ∪ B = Set.univ)

def is_prohibited (S A B : Set ℕ) : Prop :=
  ∀ k l s, (k ∈ A ∧ l ∈ A ∧ s ∈ S) ∨ (k ∈ B ∧ l ∈ B ∧ s ∈ S) → k + l ≠ s

theorem unique_fibonacci_partition :
  ∃! (A B : Set ℕ), is_partition A B ∧ is_prohibited {n | is_fibonacci n} A B :=
sorry

end NUMINAMATH_CALUDE_unique_fibonacci_partition_l1085_108522


namespace NUMINAMATH_CALUDE_mixture_replacement_l1085_108500

/-- Represents the mixture replacement problem -/
theorem mixture_replacement (initial_a initial_b replaced_amount : ℝ) : 
  initial_a = 64 →
  initial_b = initial_a / 4 →
  (initial_a - (4/5) * replaced_amount) / (initial_b - (1/5) * replaced_amount + replaced_amount) = 2/3 →
  replaced_amount = 40 :=
by
  sorry

#check mixture_replacement

end NUMINAMATH_CALUDE_mixture_replacement_l1085_108500


namespace NUMINAMATH_CALUDE_gcd_property_l1085_108524

theorem gcd_property (n : ℕ) :
  (∃ d : ℕ, d = Nat.gcd (7 * n + 5) (5 * n + 4) ∧ (d = 1 ∨ d = 3)) ∧
  (Nat.gcd (7 * n + 5) (5 * n + 4) = 3 ↔ ∃ k : ℕ, n = 3 * k + 1) :=
by sorry

end NUMINAMATH_CALUDE_gcd_property_l1085_108524


namespace NUMINAMATH_CALUDE_sunday_to_saturday_ratio_l1085_108584

/-- Tameka's cracker box sales over three days -/
structure CrackerSales where
  friday : ℕ
  saturday : ℕ
  sunday : ℕ
  total : ℕ
  h1 : friday = 40
  h2 : saturday = 2 * friday - 10
  h3 : total = friday + saturday + sunday
  h4 : total = 145

/-- The ratio of boxes sold on Sunday to boxes sold on Saturday is 1/2 -/
theorem sunday_to_saturday_ratio (sales : CrackerSales) :
  sales.sunday / sales.saturday = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sunday_to_saturday_ratio_l1085_108584


namespace NUMINAMATH_CALUDE_round_trip_speed_l1085_108568

/-- Proves that given a person's average speed for a round trip is 75 km/hr,
    and the return speed is 50% faster than the initial speed,
    the initial speed is 62.5 km/hr. -/
theorem round_trip_speed (v : ℝ) : 
  (v > 0) →                           -- Initial speed is positive
  (2 / (1 / v + 1 / (1.5 * v)) = 75)  -- Average speed is 75 km/hr
  → v = 62.5 := by sorry

end NUMINAMATH_CALUDE_round_trip_speed_l1085_108568


namespace NUMINAMATH_CALUDE_modulo_residue_sum_of_cubes_l1085_108582

theorem modulo_residue_sum_of_cubes (m : ℕ) (h : m = 17) :
  (512^3 + (6*104)^3 + (8*289)^3 + (5*68)^3) % m = 9 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_sum_of_cubes_l1085_108582


namespace NUMINAMATH_CALUDE_f_properties_l1085_108579

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -4 * x^2 else x^2 - x

theorem f_properties :
  (∃ a : ℝ, f a = -1/4 ∧ (a = -1/4 ∨ a = 1/2)) ∧
  (∃ b : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x - b = 0 ∧ f y - b = 0 ∧ f z - b = 0) →
    -1/4 < b ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1085_108579


namespace NUMINAMATH_CALUDE_toys_sold_is_eighteen_l1085_108563

/-- Given a selling price, gain equal to the cost of 3 toys, and the cost of one toy,
    calculate the number of toys sold. -/
def number_of_toys_sold (selling_price gain cost_per_toy : ℕ) : ℕ :=
  (selling_price - gain) / cost_per_toy

/-- Theorem stating that given the conditions in the problem, 
    the number of toys sold is 18. -/
theorem toys_sold_is_eighteen :
  let selling_price := 21000
  let gain := 3 * 1000
  let cost_per_toy := 1000
  number_of_toys_sold selling_price gain cost_per_toy = 18 := by
  sorry

#eval number_of_toys_sold 21000 (3 * 1000) 1000

end NUMINAMATH_CALUDE_toys_sold_is_eighteen_l1085_108563


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1085_108598

theorem right_triangle_side_length 
  (area : ℝ) 
  (side1 : ℝ) 
  (is_right_triangle : Bool) 
  (h1 : area = 8) 
  (h2 : side1 = Real.sqrt 10) 
  (h3 : is_right_triangle = true) : 
  ∃ side2 : ℝ, side2 = 1.6 * Real.sqrt 10 ∧ (1/2) * side1 * side2 = area :=
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1085_108598


namespace NUMINAMATH_CALUDE_total_cost_is_840_l1085_108523

/-- The cost of a single movie ticket in dollars -/
def movie_ticket_cost : ℚ := 30

/-- The number of movie tickets -/
def num_movie_tickets : ℕ := 8

/-- The number of football game tickets -/
def num_football_tickets : ℕ := 5

/-- The ratio of the cost of 8 movie tickets to 1 football game ticket -/
def cost_ratio : ℚ := 2

/-- The total cost of buying movie tickets and football game tickets -/
def total_cost : ℚ :=
  (num_movie_tickets : ℚ) * movie_ticket_cost +
  (num_football_tickets : ℚ) * ((num_movie_tickets : ℚ) * movie_ticket_cost / cost_ratio)

theorem total_cost_is_840 : total_cost = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_840_l1085_108523


namespace NUMINAMATH_CALUDE_bisection_termination_condition_l1085_108545

/-- Bisection method termination condition -/
theorem bisection_termination_condition 
  (f : ℝ → ℝ) (x₁ x₂ e : ℝ) (h_e : e > 0) :
  |x₁ - x₂| < e → ∃ x, x ∈ [x₁, x₂] ∧ |f x| < e :=
sorry

end NUMINAMATH_CALUDE_bisection_termination_condition_l1085_108545


namespace NUMINAMATH_CALUDE_constant_value_theorem_l1085_108585

/-- Given constants a and b, if f(x) = x^2 + 4x + 3 and f(ax + b) = x^2 + 10x + 24, then 5a - b = 2 -/
theorem constant_value_theorem (a b : ℝ) : 
  (∀ x, (x^2 + 4*x + 3 : ℝ) = ((a*x + b)^2 + 4*(a*x + b) + 3 : ℝ)) → 
  (∀ x, (x^2 + 4*x + 3 : ℝ) = (x^2 + 10*x + 24 : ℝ)) → 
  (5*a - b : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_theorem_l1085_108585


namespace NUMINAMATH_CALUDE_power_equation_solution_l1085_108592

theorem power_equation_solution : ∃ K : ℕ, 16^3 * 8^3 = 2^K ∧ K = 21 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1085_108592


namespace NUMINAMATH_CALUDE_triangle_inequality_l1085_108596

/-- Given a triangle with sides a, b, and c, and s = (a+b+c)/2, 
    if s^2 = 2ab, then s < 2a -/
theorem triangle_inequality (a b c s : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_s_def : s = (a + b + c) / 2)
  (h_s_sq : s^2 = 2*a*b) : 
  s < 2*a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1085_108596


namespace NUMINAMATH_CALUDE_sqrt_seven_inequality_l1085_108510

theorem sqrt_seven_inequality (m n : ℕ+) (h : (m : ℝ) / n < Real.sqrt 7) :
  7 - (m : ℝ)^2 / n^2 ≥ 3 / n^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_inequality_l1085_108510


namespace NUMINAMATH_CALUDE_min_bailing_rate_solution_bailing_rate_is_eight_gallons_per_minute_l1085_108521

/-- Represents the problem of finding the minimum bailing rate -/
def MinBailingRateProblem (distance : Real) (rowingSpeed : Real) (waterIntakeRate : Real) (maxWaterCapacity : Real) : Prop :=
  ∃ (bailingRate : Real),
    bailingRate ≥ 0 ∧
    (distance / rowingSpeed) * 60 * (waterIntakeRate - bailingRate) ≤ maxWaterCapacity ∧
    ∀ (r : Real), r ≥ 0 ∧ (distance / rowingSpeed) * 60 * (waterIntakeRate - r) ≤ maxWaterCapacity → r ≥ bailingRate

/-- The solution to the minimum bailing rate problem -/
theorem min_bailing_rate_solution :
  MinBailingRateProblem 1 4 10 30 → (∃ (minRate : Real), minRate = 8) :=
by
  sorry

/-- Proof that 8 gallons per minute is the minimum bailing rate required -/
theorem bailing_rate_is_eight_gallons_per_minute :
  MinBailingRateProblem 1 4 10 30 :=
by
  sorry

end NUMINAMATH_CALUDE_min_bailing_rate_solution_bailing_rate_is_eight_gallons_per_minute_l1085_108521


namespace NUMINAMATH_CALUDE_quadratic_inequality_product_l1085_108527

/-- Given a quadratic inequality x^2 + bx + c < 0 with solution set {x | 2 < x < 4}, 
    prove that bc = -48 -/
theorem quadratic_inequality_product (b c : ℝ) 
  (h : ∀ x, x^2 + b*x + c < 0 ↔ 2 < x ∧ x < 4) : b*c = -48 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_product_l1085_108527
