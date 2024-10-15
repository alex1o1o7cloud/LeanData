import Mathlib

namespace NUMINAMATH_CALUDE_binomial_sum_theorem_l3904_390477

theorem binomial_sum_theorem (A B C D : ℚ) :
  (∀ n : ℕ, n ≥ 4 → n^4 = A * Nat.choose n 4 + B * Nat.choose n 3 + C * Nat.choose n 2 + D * Nat.choose n 1) →
  A + B + C + D = 75 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_theorem_l3904_390477


namespace NUMINAMATH_CALUDE_train_speed_l3904_390415

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length time : ℝ) (h1 : length = 700) (h2 : time = 40) :
  length / time = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3904_390415


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3904_390431

theorem complex_fraction_equality (a b : ℝ) :
  (a / (1 - Complex.I)) + (b / (2 - Complex.I)) = 1 / (3 - Complex.I) →
  a = -1/5 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3904_390431


namespace NUMINAMATH_CALUDE_school_pencils_l3904_390446

theorem school_pencils (num_pens : ℕ) (pencil_cost pen_cost total_cost : ℚ) :
  num_pens = 56 ∧
  pencil_cost = 5/2 ∧
  pen_cost = 7/2 ∧
  total_cost = 291 →
  ∃ num_pencils : ℕ, num_pencils * pencil_cost + num_pens * pen_cost = total_cost ∧ num_pencils = 38 :=
by sorry

end NUMINAMATH_CALUDE_school_pencils_l3904_390446


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l3904_390471

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_ratio : (a 2 + a 3) / (a 1 + a 2) = 2)
  (h_fourth : a 4 = 8) :
  a 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l3904_390471


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3904_390461

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3904_390461


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l3904_390420

theorem negative_sixty_four_to_seven_thirds : (-64 : ℝ) ^ (7/3) = -16384 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_seven_thirds_l3904_390420


namespace NUMINAMATH_CALUDE_sum_abc_equals_33_l3904_390495

theorem sum_abc_equals_33 
  (a b c N : ℕ+) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_eq1 : N = 5*a + 3*b + 5*c)
  (h_eq2 : N = 4*a + 5*b + 4*c)
  (h_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_33_l3904_390495


namespace NUMINAMATH_CALUDE_total_distance_is_66_l3904_390433

def first_museum_distance : ℕ := 5
def second_museum_distance : ℕ := 15
def cultural_center_distance : ℕ := 10
def detour_distance : ℕ := 3

def total_distance : ℕ :=
  2 * (first_museum_distance + detour_distance) +
  2 * second_museum_distance +
  2 * cultural_center_distance

theorem total_distance_is_66 : total_distance = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_66_l3904_390433


namespace NUMINAMATH_CALUDE_train_platform_length_equality_l3904_390422

/-- Given a train and a platform with specific conditions, prove that the length of the platform equals the length of the train. -/
theorem train_platform_length_equality
  (train_speed : Real) -- Speed of the train in km/hr
  (crossing_time : Real) -- Time to cross the platform in minutes
  (train_length : Real) -- Length of the train in meters
  (h1 : train_speed = 180) -- Train speed is 180 km/hr
  (h2 : crossing_time = 1) -- Time to cross the platform is 1 minute
  (h3 : train_length = 1500) -- Length of the train is 1500 meters
  : Real := -- Length of the platform in meters
by
  sorry

#check train_platform_length_equality

end NUMINAMATH_CALUDE_train_platform_length_equality_l3904_390422


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3904_390499

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  2 * Real.sqrt 3 * (Real.sin ((A + B) / 2))^2 - Real.sin C = Real.sqrt 3 ∧
  c = Real.sqrt 3 ∧
  a = Real.sqrt 2

theorem triangle_ABC_properties {A B C a b c : ℝ} 
  (h : triangle_ABC A B C a b c) : 
  C = π / 3 ∧ 
  (1/2 * a * b * Real.sin C) = (Real.sqrt 3 + 3) / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3904_390499


namespace NUMINAMATH_CALUDE_first_quartile_of_data_set_l3904_390486

def data_set : List ℕ := [296, 301, 305, 293, 293, 305, 302, 303, 306, 294]

def first_quartile (l : List ℕ) : ℕ := sorry

theorem first_quartile_of_data_set :
  first_quartile data_set = 294 := by sorry

end NUMINAMATH_CALUDE_first_quartile_of_data_set_l3904_390486


namespace NUMINAMATH_CALUDE_ellianna_fat_served_l3904_390482

/-- The amount of fat in ounces for a herring -/
def herring_fat : ℕ := 40

/-- The amount of fat in ounces for an eel -/
def eel_fat : ℕ := 20

/-- The amount of fat in ounces for a pike -/
def pike_fat : ℕ := eel_fat + 10

/-- The number of fish of each type that Ellianna cooked and served -/
def fish_count : ℕ := 40

/-- The total amount of fat in ounces served by Ellianna -/
def total_fat : ℕ := fish_count * (herring_fat + eel_fat + pike_fat)

theorem ellianna_fat_served : total_fat = 3600 := by
  sorry

end NUMINAMATH_CALUDE_ellianna_fat_served_l3904_390482


namespace NUMINAMATH_CALUDE_root_equation_solution_l3904_390476

theorem root_equation_solution (a b c : ℕ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (∀ N : ℝ, N ≠ 1 → (N^(1/a) * (N^(1/b) * N^(3/c))^(1/a))^a = N^(15/24)) →
  c = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_solution_l3904_390476


namespace NUMINAMATH_CALUDE_carmela_money_distribution_l3904_390496

/-- Proves that Carmela needs to give $1 to each cousin for equal distribution -/
theorem carmela_money_distribution (carmela_money : ℕ) (cousin_money : ℕ) (num_cousins : ℕ) :
  carmela_money = 7 →
  cousin_money = 2 →
  num_cousins = 4 →
  let total_money := carmela_money + num_cousins * cousin_money
  let num_people := num_cousins + 1
  let equal_share := total_money / num_people
  let carmela_gives := carmela_money - equal_share
  carmela_gives / num_cousins = 1 := by
  sorry

end NUMINAMATH_CALUDE_carmela_money_distribution_l3904_390496


namespace NUMINAMATH_CALUDE_largest_fraction_l3904_390451

theorem largest_fraction :
  let fractions := [2/5, 3/7, 4/9, 7/15, 9/20, 11/25]
  ∀ x ∈ fractions, (7:ℚ)/15 ≥ x := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l3904_390451


namespace NUMINAMATH_CALUDE_smallest_possible_median_l3904_390484

def number_set (x : ℤ) : Finset ℤ := {x, 3*x, 4, 1, 6}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (λ y => y ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ y => y ≥ m)).card ≥ s.card

theorem smallest_possible_median :
  ∃ (x : ℤ), is_median 1 (number_set x) ∧
  ∀ (y : ℤ) (m : ℤ), is_median m (number_set y) → m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_median_l3904_390484


namespace NUMINAMATH_CALUDE_arcsin_neg_one_l3904_390423

theorem arcsin_neg_one : Real.arcsin (-1) = -π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_one_l3904_390423


namespace NUMINAMATH_CALUDE_qin_jiushao_triangle_area_l3904_390428

theorem qin_jiushao_triangle_area : 
  let a : ℝ := 5
  let b : ℝ := 6
  let c : ℝ := 7
  let S := Real.sqrt ((1/4) * (a^2 * b^2 - ((a^2 + b^2 - c^2)/2)^2))
  S = 6 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_qin_jiushao_triangle_area_l3904_390428


namespace NUMINAMATH_CALUDE_external_circle_radius_l3904_390492

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right angle at C
  (B.1 - C.1) * (A.1 - C.1) + (B.2 - C.2) * (A.2 - C.2) = 0 ∧
  -- Angle A is 45°
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) / Real.sqrt 2 ∧
  -- AC = 12
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 144

-- Define the external circle
def ExternalCircle (center : ℝ × ℝ) (radius : ℝ) (A B C : ℝ × ℝ) : Prop :=
  -- Circle is tangent to AB
  ((center.1 - A.1) * (B.2 - A.2) - (center.2 - A.2) * (B.1 - A.1))^2 = 
    radius^2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2) ∧
  -- Center lies on line AB
  (center.2 - A.2) * (B.1 - A.1) = (center.1 - A.1) * (B.2 - A.2)

-- Theorem statement
theorem external_circle_radius (A B C : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) :
  Triangle A B C → ExternalCircle center radius A B C → radius = 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_external_circle_radius_l3904_390492


namespace NUMINAMATH_CALUDE_initial_players_l3904_390472

theorem initial_players (initial_players new_players lives_per_player total_lives : ℕ) :
  new_players = 5 →
  lives_per_player = 3 →
  total_lives = 27 →
  (initial_players + new_players) * lives_per_player = total_lives →
  initial_players = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_players_l3904_390472


namespace NUMINAMATH_CALUDE_expression_value_l3904_390401

theorem expression_value : 
  (150^2 - 13^2) / (90^2 - 17^2) * ((90-17)*(90+17)) / ((150-13)*(150+13)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3904_390401


namespace NUMINAMATH_CALUDE_last_eight_digits_of_product_l3904_390405

theorem last_eight_digits_of_product : ∃ n : ℕ, 
  11 * 101 * 1001 * 10001 * 100001 * 1000001 * 111 ≡ 19754321 [MOD 100000000] :=
by sorry

end NUMINAMATH_CALUDE_last_eight_digits_of_product_l3904_390405


namespace NUMINAMATH_CALUDE_patty_weeks_without_chores_l3904_390432

/-- Calculates the number of weeks Patty can go without doing chores --/
def weeks_without_chores (
  cookies_per_chore : ℕ) 
  (chores_per_kid_per_week : ℕ) 
  (money_available : ℕ) 
  (cookies_per_pack : ℕ) 
  (cost_per_pack : ℕ) 
  (num_siblings : ℕ) : ℕ :=
  let packs_bought := money_available / cost_per_pack
  let total_cookies := packs_bought * cookies_per_pack
  let cookies_per_sibling_per_week := chores_per_kid_per_week * cookies_per_chore
  let cookies_needed_per_week := cookies_per_sibling_per_week * num_siblings
  total_cookies / cookies_needed_per_week

theorem patty_weeks_without_chores :
  weeks_without_chores 3 4 15 24 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_patty_weeks_without_chores_l3904_390432


namespace NUMINAMATH_CALUDE_expression_value_l3904_390442

theorem expression_value (a b : ℤ) (ha : a = -3) (hb : b = 2) :
  -a - b^3 + a*b = -11 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3904_390442


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l3904_390414

/-- Given a circle with center (4, 6) and one endpoint of a diameter at (2, 1),
    prove that the other endpoint of the diameter is at (6, 11). -/
theorem circle_diameter_endpoint (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : 
  P = (4, 6) →  -- Center of the circle
  A = (2, 1) →  -- One endpoint of the diameter
  (P.1 - A.1 = B.1 - P.1 ∧ P.2 - A.2 = B.2 - P.2) →  -- B is symmetric to A with respect to P
  B = (6, 11) :=  -- The other endpoint of the diameter
by sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l3904_390414


namespace NUMINAMATH_CALUDE_no_rational_roots_l3904_390408

def polynomial (x : ℚ) : ℚ :=
  3 * x^5 + 4 * x^4 - 5 * x^3 - 15 * x^2 + 7 * x + 3

theorem no_rational_roots :
  ∀ (x : ℚ), polynomial x ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_rational_roots_l3904_390408


namespace NUMINAMATH_CALUDE_gravel_density_l3904_390403

/-- Proves that the density of gravel is approximately 267 kg/m³ given the conditions of the bucket problem. -/
theorem gravel_density (bucket_volume : ℝ) (additional_water : ℝ) (full_bucket_weight : ℝ) (empty_bucket_weight : ℝ) 
  (h1 : bucket_volume = 12)
  (h2 : additional_water = 3)
  (h3 : full_bucket_weight = 28)
  (h4 : empty_bucket_weight = 1)
  (h5 : ∀ x, x > 0 → x * 1 = x) -- 1 liter of water weighs 1 kg
  : ∃ (density : ℝ), abs (density - 267) < 1 ∧ 
    density = (full_bucket_weight - empty_bucket_weight - additional_water) / 
              (bucket_volume - additional_water) * 1000 := by
  sorry

end NUMINAMATH_CALUDE_gravel_density_l3904_390403


namespace NUMINAMATH_CALUDE_integer_division_l3904_390438

theorem integer_division (x : ℤ) :
  (∃ k : ℤ, (5 * x + 2) = 17 * k) ↔ (∃ m : ℤ, x = 17 * m + 3) :=
by sorry

end NUMINAMATH_CALUDE_integer_division_l3904_390438


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l3904_390430

theorem fraction_sum_theorem : 
  (1 : ℚ) / 15 + 2 / 15 + 3 / 15 + 4 / 15 + 5 / 15 + 
  6 / 15 + 7 / 15 + 8 / 15 + 9 / 15 + 46 / 15 = 91 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l3904_390430


namespace NUMINAMATH_CALUDE_evaluate_expression_l3904_390465

theorem evaluate_expression (x z : ℤ) (hx : x = 4) (hz : z = -2) :
  z * (z - 4 * x) = 36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3904_390465


namespace NUMINAMATH_CALUDE_line_slope_is_two_l3904_390487

/-- Given a line with y-intercept 2 and passing through the point (498, 998), its slope is 2 -/
theorem line_slope_is_two (f : ℝ → ℝ) (h1 : f 0 = 2) (h2 : f 498 = 998) :
  (f 498 - f 0) / (498 - 0) = 2 := by
sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l3904_390487


namespace NUMINAMATH_CALUDE_triangle_side_range_l3904_390479

theorem triangle_side_range (a b c : ℝ) : 
  -- Triangle ABC is acute
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  -- Side lengths form an arithmetic sequence
  (b - a = c - b) ∧
  -- Sum of squares of sides equals 21
  a^2 + b^2 + c^2 = 21 →
  -- Range of b
  2 * Real.sqrt 42 / 5 < b ∧ b ≤ Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l3904_390479


namespace NUMINAMATH_CALUDE_equation_rewrite_l3904_390426

/-- Given an equation with roots α and β, prove that a related equation can be rewritten in terms of α, β, and a constant k. -/
theorem equation_rewrite (a b c d α β : ℝ) (hα : α = (a * α + b) / (c * α + d)) (hβ : β = (a * β + b) / (c * β + d)) :
  ∃ k : ℝ, ∀ y z : ℝ, y = (a * z + b) / (c * z + d) →
    (y - α) / (y - β) = k * (z - α) / (z - β) ∧ k = (c * β + d) / (c * α + d) := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l3904_390426


namespace NUMINAMATH_CALUDE_min_distance_sum_l3904_390475

theorem min_distance_sum (x : ℝ) : 
  ∃ (min : ℝ) (x_min : ℝ), 
    min = |x_min - 2| + |x_min - 4| + |x_min - 10| ∧
    min = 8 ∧ 
    x_min = 4 ∧
    ∀ y : ℝ, |y - 2| + |y - 4| + |y - 10| ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l3904_390475


namespace NUMINAMATH_CALUDE_symmetric_lines_coefficient_sum_l3904_390445

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + 2*y - 3 = 0
def line2 (a b x y : ℝ) : Prop := a*x + 4*y + b = 0

-- Define the point A
def point_A : ℝ × ℝ := (1, 0)

-- Define symmetry with respect to a point
def symmetric_wrt (p : ℝ × ℝ) (l1 l2 : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x y : ℝ), l1 x y ↔ l2 (2*p.1 - x) (2*p.2 - y)

-- Theorem statement
theorem symmetric_lines_coefficient_sum (a b : ℝ) :
  symmetric_wrt point_A (line1) (line2 a b) →
  a + b = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_lines_coefficient_sum_l3904_390445


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3904_390443

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3904_390443


namespace NUMINAMATH_CALUDE_erroneous_product_l3904_390449

/-- Given two positive integers a and b, where a is a two-digit number,
    if reversing the digits of a and multiplying by b, then adding 2, results in 240,
    then the actual product of a and b is 301. -/
theorem erroneous_product (a b : ℕ) : 
  a > 9 ∧ a < 100 →  -- a is a two-digit number
  b > 0 →  -- b is positive
  (((a % 10) * 10 + (a / 10)) * b + 2 = 240) →  -- erroneous calculation
  a * b = 301 := by
sorry

end NUMINAMATH_CALUDE_erroneous_product_l3904_390449


namespace NUMINAMATH_CALUDE_simplify_expression_l3904_390427

theorem simplify_expression (x y : ℝ) : 
  (2 * x + 20) + (150 * x + 30) + y = 152 * x + 50 + y := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3904_390427


namespace NUMINAMATH_CALUDE_train_speed_proof_l3904_390485

/-- Proves that the speed of each train is 54 km/hr given the problem conditions -/
theorem train_speed_proof (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 120) (h2 : crossing_time = 8) : 
  let relative_speed := (2 * train_length) / crossing_time
  let train_speed_ms := relative_speed / 2
  let train_speed_kmh := train_speed_ms * 3.6
  train_speed_kmh = 54 := by
sorry


end NUMINAMATH_CALUDE_train_speed_proof_l3904_390485


namespace NUMINAMATH_CALUDE_final_number_calculation_l3904_390483

theorem final_number_calculation : 
  let initial_number : ℕ := 9
  let doubled := initial_number * 2
  let added_13 := doubled + 13
  let final_number := added_13 * 3
  final_number = 93 := by sorry

end NUMINAMATH_CALUDE_final_number_calculation_l3904_390483


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3904_390424

/-- 
Given a geometric sequence with positive terms, if the sum of the first n terms is 3
and the sum of the first 3n terms is 21, then the sum of the first 2n terms is 9.
-/
theorem geometric_sequence_sum (n : ℕ) (a : ℝ) (r : ℝ) 
  (h_positive : ∀ k, a * r ^ k > 0)
  (h_sum_n : (a * (1 - r^n)) / (1 - r) = 3)
  (h_sum_3n : (a * (1 - r^(3*n))) / (1 - r) = 21) :
  (a * (1 - r^(2*n))) / (1 - r) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3904_390424


namespace NUMINAMATH_CALUDE_five_digit_sum_contains_zero_l3904_390469

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digits_differ_by_two (a b : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 e1 e2 e3 e4 e5 : ℕ),
    a = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    b = e1 * 10000 + e2 * 1000 + e3 * 100 + e4 * 10 + e5 ∧
    ({d1, d2, d3, d4, d5} : Finset ℕ) = {e1, e2, e3, e4, e5} ∧
    (d1 = e1 ∧ d2 = e2 ∧ d4 = e4 ∧ d5 = e5) ∨
    (d1 = e1 ∧ d2 = e2 ∧ d3 = e3 ∧ d5 = e5) ∨
    (d1 = e1 ∧ d2 = e2 ∧ d3 = e3 ∧ d4 = e4)

def contains_zero (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 : ℕ),
    n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5 ∧
    (d1 = 0 ∨ d2 = 0 ∨ d3 = 0 ∨ d4 = 0 ∨ d5 = 0)

theorem five_digit_sum_contains_zero (a b : ℕ) :
  is_five_digit a → is_five_digit b → digits_differ_by_two a b → a + b = 111111 →
  contains_zero a ∨ contains_zero b :=
sorry

end NUMINAMATH_CALUDE_five_digit_sum_contains_zero_l3904_390469


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3904_390481

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) 
  (h : 3 • a + (3/5 : ℝ) • (b - x) = b) : 
  x = 5 • a - (2/3 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3904_390481


namespace NUMINAMATH_CALUDE_intersection_point_l3904_390404

/-- A parabola in the xy-plane defined by y^2 - 4y + x = 6 -/
def parabola (x y : ℝ) : Prop := y^2 - 4*y + x = 6

/-- A vertical line in the xy-plane defined by x = k -/
def vertical_line (k x : ℝ) : Prop := x = k

/-- The condition for a quadratic equation ay^2 + by + c = 0 to have exactly one solution -/
def has_unique_solution (a b c : ℝ) : Prop := b^2 - 4*a*c = 0

theorem intersection_point (k : ℝ) : 
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ vertical_line k p.1) ↔ k = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3904_390404


namespace NUMINAMATH_CALUDE_sum_of_bases_is_nineteen_l3904_390413

/-- Represents a repeating decimal in a given base -/
def RepeatingDecimal (numerator : ℕ) (denominator : ℕ) (base : ℕ) : Prop :=
  ∃ (k : ℕ), (base ^ k * numerator) % denominator = numerator

/-- The main theorem -/
theorem sum_of_bases_is_nineteen (R₁ R₂ : ℕ) :
  R₁ > 1 ∧ R₂ > 1 ∧
  RepeatingDecimal 5 11 R₁ ∧
  RepeatingDecimal 6 11 R₁ ∧
  RepeatingDecimal 5 13 R₂ ∧
  RepeatingDecimal 8 13 R₂ →
  R₁ + R₂ = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_is_nineteen_l3904_390413


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3904_390400

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 1) ↔ (∃ x : ℝ, x < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3904_390400


namespace NUMINAMATH_CALUDE_arccos_cos_eight_l3904_390459

theorem arccos_cos_eight (h : 0 ≤ 8 - 2 * Real.pi ∧ 8 - 2 * Real.pi < Real.pi) :
  Real.arccos (Real.cos 8) = 8 - 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eight_l3904_390459


namespace NUMINAMATH_CALUDE_bill_calculation_l3904_390467

theorem bill_calculation (a b c d : ℝ) 
  (h1 : (a - b) + c - d = 19) 
  (h2 : a - b - c - d = 9) : 
  a - b = 14 := by
sorry

end NUMINAMATH_CALUDE_bill_calculation_l3904_390467


namespace NUMINAMATH_CALUDE_flowerbed_length_difference_l3904_390440

theorem flowerbed_length_difference (width length : ℝ) : 
  width = 4 →
  2 * length + 2 * width = 22 →
  2 * width - length = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_flowerbed_length_difference_l3904_390440


namespace NUMINAMATH_CALUDE_boat_downstream_time_l3904_390425

/-- Proves that a boat traveling downstream takes 1 hour to cover 45 km,
    given its speed in still water and the stream's speed. -/
theorem boat_downstream_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 40)
  (h2 : stream_speed = 5)
  (h3 : distance = 45) :
  distance / (boat_speed + stream_speed) = 1 :=
by sorry

end NUMINAMATH_CALUDE_boat_downstream_time_l3904_390425


namespace NUMINAMATH_CALUDE_least_integer_with_specific_divisibility_l3904_390412

theorem least_integer_with_specific_divisibility : ∃ (n : ℕ), 
  (∀ (k : ℕ), k ≤ 28 → k ∣ n) ∧ 
  (30 ∣ n) ∧ 
  ¬(29 ∣ n) ∧
  (∀ (m : ℕ), m < n → ¬((∀ (k : ℕ), k ≤ 28 → k ∣ m) ∧ (30 ∣ m) ∧ ¬(29 ∣ m))) ∧
  n = 232792560 := by
sorry

end NUMINAMATH_CALUDE_least_integer_with_specific_divisibility_l3904_390412


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3904_390454

-- Define the sets A and B
def A : Set ℝ := { x | 2 ≤ x ∧ x < 4 }
def B : Set ℝ := { x | 3 * x - 7 ≥ 8 - 2 * x }

-- State the theorem
theorem union_of_A_and_B : A ∪ B = { x | x ≥ 2 } := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3904_390454


namespace NUMINAMATH_CALUDE_decimal_41_to_binary_l3904_390455

-- Define a function to convert decimal to binary
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

-- Theorem statement
theorem decimal_41_to_binary :
  decimalToBinary 41 = [1, 0, 1, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_41_to_binary_l3904_390455


namespace NUMINAMATH_CALUDE_log_expression_simplification_l3904_390458

theorem log_expression_simplification 
  (p q r s x y : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (hx : x > 0) (hy : y > 0) : 
  Real.log (p^2 / q) + Real.log (q^3 / r) + Real.log (r^2 / s) - Real.log (p^2 * y / (s^3 * x)) 
  = Real.log (q^2 * r * x * s^2 / y) := by
  sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l3904_390458


namespace NUMINAMATH_CALUDE_inequalities_given_sum_positive_l3904_390407

theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_sum_positive_l3904_390407


namespace NUMINAMATH_CALUDE_absolute_value_inequality_implies_a_bound_l3904_390406

theorem absolute_value_inequality_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, |x + 2| - |x - 1| ≥ a^3 - 4*a^2 - 3) →
  a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_implies_a_bound_l3904_390406


namespace NUMINAMATH_CALUDE_coin_sequences_ten_l3904_390444

/-- The number of distinct sequences when flipping a coin n times -/
def coin_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences when flipping a coin 10 times is 1024 -/
theorem coin_sequences_ten : coin_sequences 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_coin_sequences_ten_l3904_390444


namespace NUMINAMATH_CALUDE_renovation_material_sum_l3904_390436

/-- The amount of sand required for the renovation project in truck-loads -/
def sand : ℚ := 0.16666666666666666

/-- The amount of dirt required for the renovation project in truck-loads -/
def dirt : ℚ := 0.3333333333333333

/-- The amount of cement required for the renovation project in truck-loads -/
def cement : ℚ := 0.16666666666666666

/-- The total amount of material required for the renovation project in truck-loads -/
def total_material : ℚ := 0.6666666666666666

/-- Theorem stating that the sum of sand, dirt, and cement equals the total material required -/
theorem renovation_material_sum :
  sand + dirt + cement = total_material := by sorry

end NUMINAMATH_CALUDE_renovation_material_sum_l3904_390436


namespace NUMINAMATH_CALUDE_arrangements_count_l3904_390468

/-- The number of ways to arrange 2 objects out of 2 positions -/
def A_2_2 : ℕ := 2

/-- The number of ways to arrange 2 objects out of 3 positions -/
def A_3_2 : ℕ := 6

/-- The number of ways to bind A and B together -/
def bind_AB : ℕ := 2

/-- The total number of people -/
def total_people : ℕ := 5

/-- Theorem: The number of arrangements of 5 people where A and B must stand next to each other,
    and C and D cannot stand next to each other, is 24. -/
theorem arrangements_count : 
  bind_AB * A_2_2 * A_3_2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l3904_390468


namespace NUMINAMATH_CALUDE_platform_length_l3904_390437

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 8 seconds to cross a signal pole, prove that the length of the platform is 1162.5 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 8) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 1162.5 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3904_390437


namespace NUMINAMATH_CALUDE_sector_area_l3904_390466

/-- The area of a circular sector with central angle 120° and radius 3/2 is 3/4 π. -/
theorem sector_area (angle : Real) (radius : Real) : 
  angle = 120 * π / 180 → radius = 3 / 2 → 
  (angle / (2 * π)) * π * radius^2 = 3 / 4 * π := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l3904_390466


namespace NUMINAMATH_CALUDE_daily_water_intake_l3904_390491

-- Define the given conditions
def daily_soda_cans : ℕ := 5
def ounces_per_can : ℕ := 12
def weekly_total_fluid : ℕ := 868

-- Define the daily soda intake in ounces
def daily_soda_ounces : ℕ := daily_soda_cans * ounces_per_can

-- Define the weekly soda intake in ounces
def weekly_soda_ounces : ℕ := daily_soda_ounces * 7

-- Define the weekly water intake in ounces
def weekly_water_ounces : ℕ := weekly_total_fluid - weekly_soda_ounces

-- Theorem to prove
theorem daily_water_intake : weekly_water_ounces / 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_daily_water_intake_l3904_390491


namespace NUMINAMATH_CALUDE_max_distance_line_ellipse_intersection_l3904_390456

/-- The maximum distance between two intersection points of a line and an ellipse -/
theorem max_distance_line_ellipse_intersection :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + 4*y^2 = 4}
  let line (m : ℝ) := {(x, y) : ℝ × ℝ | y = x + m}
  let intersection (m : ℝ) := {p : ℝ × ℝ | p ∈ ellipse ∧ p ∈ line m}
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (m : ℝ), ∀ (p q : ℝ × ℝ), p ∈ intersection m → q ∈ intersection m → p ≠ q →
    distance p q ≤ (4/5) * Real.sqrt 10 ∧
    ∃ (m' : ℝ) (p' q' : ℝ × ℝ), p' ∈ intersection m' ∧ q' ∈ intersection m' ∧ p' ≠ q' ∧
      distance p' q' = (4/5) * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_max_distance_line_ellipse_intersection_l3904_390456


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l3904_390418

theorem koala_fiber_consumption (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_consumed : ℝ) :
  absorption_rate = 0.25 →
  absorbed_amount = 10.5 →
  absorbed_amount = absorption_rate * total_consumed →
  total_consumed = 42 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_consumption_l3904_390418


namespace NUMINAMATH_CALUDE_percentage_of_female_officers_on_duty_l3904_390434

def total_officers_on_duty : ℕ := 160
def total_female_officers : ℕ := 500

def female_officers_on_duty : ℕ := total_officers_on_duty / 2

def percentage_on_duty : ℚ := (female_officers_on_duty : ℚ) / total_female_officers * 100

theorem percentage_of_female_officers_on_duty :
  percentage_on_duty = 16 := by sorry

end NUMINAMATH_CALUDE_percentage_of_female_officers_on_duty_l3904_390434


namespace NUMINAMATH_CALUDE_sum_interior_angles_formula_l3904_390441

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: For a polygon with n sides (n > 2), the sum of interior angles is 180° × (n-2) -/
theorem sum_interior_angles_formula {n : ℕ} (h : n > 2) :
  sum_interior_angles n = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_formula_l3904_390441


namespace NUMINAMATH_CALUDE_zeros_of_f_with_fixed_points_range_of_b_with_no_fixed_points_l3904_390435

-- Define the function f(x)
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Theorem 1
theorem zeros_of_f_with_fixed_points (b c : ℝ) :
  (f b c (-3) = -3) ∧ (f b c 2 = 2) →
  (∃ x : ℝ, f b c x = 0) ∧ 
  (∀ x : ℝ, f b c x = 0 ↔ (x = -1 + Real.sqrt 7 ∨ x = -1 - Real.sqrt 7)) :=
sorry

-- Theorem 2
theorem range_of_b_with_no_fixed_points :
  (∀ b : ℝ, ∀ x : ℝ, f b (b^2/4) x ≠ x) →
  (∀ b : ℝ, (b < -1 ∨ b > 1/3) ↔ (∀ x : ℝ, f b (b^2/4) x ≠ x)) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_with_fixed_points_range_of_b_with_no_fixed_points_l3904_390435


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3904_390464

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 600)
  (h2 : profit_percentage = 25) :
  let cost_price := selling_price / (1 + profit_percentage / 100)
  cost_price = 480 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3904_390464


namespace NUMINAMATH_CALUDE_sqrt_of_16_l3904_390480

theorem sqrt_of_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_16_l3904_390480


namespace NUMINAMATH_CALUDE_inequalities_and_minimum_value_l3904_390462

theorem inequalities_and_minimum_value :
  (∀ a b, a > b ∧ b > 0 → (1 / a : ℝ) < (1 / b)) ∧
  (∀ a b, a > b ∧ b > 0 → a - 1 / a > b - 1 / b) ∧
  (∀ a b, a > b ∧ b > 0 → (2 * a + b) / (a + 2 * b) < a / b) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 → 2 / a + 1 / b ≥ 9 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ 2 / a + 1 / b = 9) :=
by sorry


end NUMINAMATH_CALUDE_inequalities_and_minimum_value_l3904_390462


namespace NUMINAMATH_CALUDE_upstream_distance_l3904_390450

/-- Proves that given a man who swims downstream 30 km in 6 hours and upstream for 6 hours, 
    with a speed of 4 km/h in still water, the distance he swims upstream is 18 km. -/
theorem upstream_distance 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : downstream_distance = 30)
  (h2 : downstream_time = 6)
  (h3 : upstream_time = 6)
  (h4 : still_water_speed = 4) : 
  ∃ upstream_distance : ℝ, upstream_distance = 18 := by
  sorry


end NUMINAMATH_CALUDE_upstream_distance_l3904_390450


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3904_390411

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 > 0 →
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 16 →
  a 3 + a 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3904_390411


namespace NUMINAMATH_CALUDE_q_work_time_l3904_390452

-- Define the work rates and total work
variable (W : ℝ) -- Total work
variable (Wp Wq Wr : ℝ) -- Work rates of p, q, and r

-- Define the conditions
axiom condition1 : Wp = Wq + Wr
axiom condition2 : Wp + Wq = W / 10
axiom condition3 : Wr = W / 60

-- Theorem to prove
theorem q_work_time : Wq = W / 24 := by
  sorry


end NUMINAMATH_CALUDE_q_work_time_l3904_390452


namespace NUMINAMATH_CALUDE_value_of_expression_l3904_390478

theorem value_of_expression (a b : ℤ) (A B : ℤ) 
  (h1 : A = 3 * b^2 - 2 * a^2)
  (h2 : B = a * b - 2 * b^2 - a^2)
  (h3 : a = 2)
  (h4 : b = -1) :
  A - 2 * B = 11 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3904_390478


namespace NUMINAMATH_CALUDE_toys_sold_is_eighteen_l3904_390439

/-- The number of toys sold by a man, given the selling price, gain, and cost price per toy. -/
def number_of_toys_sold (selling_price gain cost_per_toy : ℕ) : ℕ :=
  (selling_price - gain) / cost_per_toy

/-- Theorem stating that the number of toys sold is 18 under the given conditions. -/
theorem toys_sold_is_eighteen :
  let selling_price : ℕ := 25200
  let cost_per_toy : ℕ := 1200
  let gain : ℕ := 3 * cost_per_toy
  number_of_toys_sold selling_price gain cost_per_toy = 18 := by
sorry

#eval number_of_toys_sold 25200 (3 * 1200) 1200

end NUMINAMATH_CALUDE_toys_sold_is_eighteen_l3904_390439


namespace NUMINAMATH_CALUDE_special_triangle_sides_l3904_390493

/-- A triangle with consecutive integer side lengths and a median perpendicular to an angle bisector -/
structure SpecialTriangle where
  a : ℕ
  has_consecutive_sides : a > 0
  median_perpendicular_to_bisector : Bool

/-- The sides of a special triangle are 2, 3, and 4 -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 2 := by
  sorry

#check special_triangle_sides

end NUMINAMATH_CALUDE_special_triangle_sides_l3904_390493


namespace NUMINAMATH_CALUDE_jessie_weight_calculation_l3904_390447

/-- Calculates the initial weight given the current weight and weight lost -/
def initial_weight (current_weight weight_lost : ℝ) : ℝ :=
  current_weight + weight_lost

/-- Theorem: If Jessie's current weight is 27 kg and she lost 10 kg, her initial weight was 37 kg -/
theorem jessie_weight_calculation :
  let current_weight : ℝ := 27
  let weight_lost : ℝ := 10
  initial_weight current_weight weight_lost = 37 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_calculation_l3904_390447


namespace NUMINAMATH_CALUDE_sperm_genotypes_l3904_390421

-- Define the possible alleles
inductive Allele
| A
| a
| Xb
| Y

-- Define a genotype as a list of alleles
def Genotype := List Allele

-- Define the initial spermatogonial cell genotype
def initialGenotype : Genotype := [Allele.A, Allele.a, Allele.Xb, Allele.Y]

-- Define the genotype of the abnormal sperm
def abnormalSperm : Genotype := [Allele.A, Allele.A, Allele.a, Allele.Xb]

-- Define the function to check if a list of genotypes is valid
def isValidResult (sperm1 sperm2 sperm3 : Genotype) : Prop :=
  sperm1 = [Allele.a, Allele.Xb] ∧
  sperm2 = [Allele.Y] ∧
  sperm3 = [Allele.Y]

-- State the theorem
theorem sperm_genotypes (initialCell : Genotype) (abnormalSperm : Genotype) :
  initialCell = initialGenotype →
  abnormalSperm = abnormalSperm →
  ∃ (sperm1 sperm2 sperm3 : Genotype), isValidResult sperm1 sperm2 sperm3 :=
sorry

end NUMINAMATH_CALUDE_sperm_genotypes_l3904_390421


namespace NUMINAMATH_CALUDE_black_rhinos_count_l3904_390409

/-- The number of white rhinos -/
def num_white_rhinos : ℕ := 7

/-- The weight of each white rhino in pounds -/
def weight_white_rhino : ℕ := 5100

/-- The weight of each black rhino in pounds -/
def weight_black_rhino : ℕ := 2000

/-- The total weight of all rhinos in pounds -/
def total_weight : ℕ := 51700

/-- The number of black rhinos -/
def num_black_rhinos : ℕ := (total_weight - num_white_rhinos * weight_white_rhino) / weight_black_rhino

theorem black_rhinos_count : num_black_rhinos = 8 := by sorry

end NUMINAMATH_CALUDE_black_rhinos_count_l3904_390409


namespace NUMINAMATH_CALUDE_bert_toy_phones_l3904_390419

/-- 
Proves that Bert sold 8 toy phones given the conditions of the problem.
-/
theorem bert_toy_phones :
  ∀ (bert_phones : ℕ),
  (18 * bert_phones = 20 * 7 + 4) →
  bert_phones = 8 := by
  sorry

end NUMINAMATH_CALUDE_bert_toy_phones_l3904_390419


namespace NUMINAMATH_CALUDE_fraction_simplification_l3904_390498

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y) / (1 / x) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3904_390498


namespace NUMINAMATH_CALUDE_seating_theorem_l3904_390463

/-- The number of ways to arrange n objects from m choices --/
def permutation (n m : ℕ) : ℕ := 
  if n > m then 0
  else Nat.factorial m / Nat.factorial (m - n)

/-- The number of ways four people can sit in a row of five chairs --/
def seating_arrangements : ℕ := permutation 4 5

theorem seating_theorem : seating_arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l3904_390463


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3904_390417

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating the smallest possible perimeter of a scalene triangle
    with prime side lengths greater than 3 and prime perimeter -/
theorem smallest_prime_perimeter_scalene_triangle :
  ∃ (a b c : ℕ),
    a < b ∧ b < c ∧
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    a > 3 ∧ b > 3 ∧ c > 3 ∧
    isValidTriangle a b c ∧
    isPrime (a + b + c) ∧
    (∀ (x y z : ℕ),
      x < y ∧ y < z ∧
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      x > 3 ∧ y > 3 ∧ z > 3 ∧
      isValidTriangle x y z ∧
      isPrime (x + y + z) →
      a + b + c ≤ x + y + z) ∧
    a + b + c = 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3904_390417


namespace NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l3904_390488

/-- Given two planes α and β with normal vectors (x, 1, -2) and (-1, y, 1/2) respectively,
    if α is parallel to β, then x + y = 15/4 -/
theorem parallel_planes_normal_vectors (x y : ℝ) :
  let n1 : ℝ × ℝ × ℝ := (x, 1, -2)
  let n2 : ℝ × ℝ × ℝ := (-1, y, 1/2)
  (∃ (k : ℝ), n1 = k • n2) →
  x + y = 15/4 := by
sorry

end NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l3904_390488


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l3904_390402

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence with the given properties, the ninth term is 35. -/
theorem ninth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third_term : a 3 = 23)
  (h_sixth_term : a 6 = 29) :
  a 9 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l3904_390402


namespace NUMINAMATH_CALUDE_new_ratio_is_7_to_5_l3904_390457

/-- Represents the ratio of toddlers to infants -/
structure Ratio :=
  (toddlers : ℕ)
  (infants : ℕ)

def initial_ratio : Ratio := ⟨7, 3⟩
def toddler_count : ℕ := 42
def new_infants : ℕ := 12

def calculate_new_ratio (r : Ratio) (t : ℕ) (n : ℕ) : Ratio :=
  let initial_infants := t * r.infants / r.toddlers
  ⟨t, initial_infants + n⟩

theorem new_ratio_is_7_to_5 :
  let new_ratio := calculate_new_ratio initial_ratio toddler_count new_infants
  ∃ (k : ℕ), k > 0 ∧ new_ratio.toddlers = 7 * k ∧ new_ratio.infants = 5 * k :=
sorry

end NUMINAMATH_CALUDE_new_ratio_is_7_to_5_l3904_390457


namespace NUMINAMATH_CALUDE_product_equals_243_l3904_390448

theorem product_equals_243 : 
  (1 / 3 : ℚ) * 9 * (1 / 27 : ℚ) * 81 * (1 / 243 : ℚ) * 729 * (1 / 2187 : ℚ) * 6561 * (1 / 19683 : ℚ) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l3904_390448


namespace NUMINAMATH_CALUDE_number_of_female_democrats_l3904_390497

theorem number_of_female_democrats
  (total : ℕ)
  (h_total : total = 780)
  (female : ℕ)
  (male : ℕ)
  (h_sum : female + male = total)
  (female_democrats : ℕ)
  (male_democrats : ℕ)
  (h_female_dem : female_democrats = female / 2)
  (h_male_dem : male_democrats = male / 4)
  (h_total_dem : female_democrats + male_democrats = total / 3) :
  female_democrats = 130 := by
sorry

end NUMINAMATH_CALUDE_number_of_female_democrats_l3904_390497


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l3904_390474

/-- Represents the number of stamps Simon received from each friend -/
structure FriendStamps where
  x1 : ℕ
  x2 : ℕ
  x3 : ℕ
  x4 : ℕ
  x5 : ℕ

/-- Theorem representing the stamp collection problem -/
theorem stamp_collection_problem 
  (initial_stamps final_stamps : ℕ) 
  (friend_stamps : FriendStamps) : 
  initial_stamps = 34 →
  final_stamps = 61 →
  friend_stamps.x1 = 12 →
  friend_stamps.x3 = 21 →
  friend_stamps.x5 = 10 →
  friend_stamps.x1 + friend_stamps.x2 + friend_stamps.x3 + 
  friend_stamps.x4 + friend_stamps.x5 = final_stamps - initial_stamps :=
by
  sorry

#check stamp_collection_problem

end NUMINAMATH_CALUDE_stamp_collection_problem_l3904_390474


namespace NUMINAMATH_CALUDE_triangle_similarity_equivalence_l3904_390489

theorem triangle_similarity_equivalence 
  (a b c a₁ b₁ c₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ k : ℝ, k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁) ↔ 
  (Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) = 
   Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_equivalence_l3904_390489


namespace NUMINAMATH_CALUDE_min_distance_four_points_l3904_390490

/-- Given four points A, B, C, and D on a line, where the distances between consecutive
    points are AB = 10, BC = 4, and CD = 3, the minimum possible distance between A and D is 3. -/
theorem min_distance_four_points (A B C D : ℝ) : 
  |B - A| = 10 → |C - B| = 4 → |D - C| = 3 → 
  (∃ (A' B' C' D' : ℝ), |B' - A'| = 10 ∧ |C' - B'| = 4 ∧ |D' - C'| = 3 ∧ 
    ∀ (X Y Z W : ℝ), |Y - X| = 10 → |Z - Y| = 4 → |W - Z| = 3 → |W - X| ≥ |D' - A'|) →
  (∃ (A₀ B₀ C₀ D₀ : ℝ), |B₀ - A₀| = 10 ∧ |C₀ - B₀| = 4 ∧ |D₀ - C₀| = 3 ∧ |D₀ - A₀| = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_four_points_l3904_390490


namespace NUMINAMATH_CALUDE_complex_powers_sum_l3904_390470

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_powers_sum : i^245 + i^246 + i^247 + i^248 + i^249 = i := by sorry

end NUMINAMATH_CALUDE_complex_powers_sum_l3904_390470


namespace NUMINAMATH_CALUDE_triangle_angle_identity_l3904_390473

theorem triangle_angle_identity (α β γ : Real) (h : α + β + γ = π) :
  (Real.sin β)^2 + (Real.sin γ)^2 - 2 * (Real.sin β) * (Real.sin γ) * (Real.cos α) = (Real.sin α)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_identity_l3904_390473


namespace NUMINAMATH_CALUDE_cat_arrangement_count_l3904_390460

/-- Represents the number of cat cages -/
def num_cages : ℕ := 5

/-- Represents the number of golden tabby cats -/
def num_golden : ℕ := 3

/-- Represents the number of silver tabby cats -/
def num_silver : ℕ := 4

/-- Represents the number of ragdoll cats -/
def num_ragdoll : ℕ := 1

/-- Represents the number of ways to arrange silver tabby cats in pairs -/
def silver_arrangements : ℕ := 3

/-- Represents the total number of units to arrange (golden group, 2 silver pairs, ragdoll) -/
def total_units : ℕ := 4

/-- Theorem stating the number of possible arrangements -/
theorem cat_arrangement_count :
  (Nat.choose num_cages total_units) * Nat.factorial total_units * silver_arrangements = 360 := by
  sorry

end NUMINAMATH_CALUDE_cat_arrangement_count_l3904_390460


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3904_390494

theorem cube_surface_area_increase :
  ∀ (s : ℝ), s > 0 →
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.4 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.96 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3904_390494


namespace NUMINAMATH_CALUDE_sophie_donuts_result_l3904_390429

def sophie_donuts (budget : ℝ) (box_cost : ℝ) (discount_rate : ℝ) 
                   (boxes_bought : ℕ) (donuts_per_box : ℕ) 
                   (boxes_to_mom : ℕ) (donuts_to_sister : ℕ) : ℝ × ℕ :=
  let total_cost := box_cost * boxes_bought
  let discounted_cost := total_cost * (1 - discount_rate)
  let total_donuts := boxes_bought * donuts_per_box
  let donuts_given := boxes_to_mom * donuts_per_box + donuts_to_sister
  let donuts_left := total_donuts - donuts_given
  (discounted_cost, donuts_left)

theorem sophie_donuts_result : 
  sophie_donuts 50 12 0.1 4 12 1 6 = (43.2, 30) :=
by sorry

end NUMINAMATH_CALUDE_sophie_donuts_result_l3904_390429


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l3904_390453

/-- Represents a number in a given base -/
def BaseRepresentation (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

/-- The problem statement -/
theorem least_sum_of_bases :
  ∃ (c d : Nat),
    c > 0 ∧ d > 0 ∧
    BaseRepresentation [5, 8] c = BaseRepresentation [8, 5] d ∧
    (∀ (c' d' : Nat),
      c' > 0 → d' > 0 →
      BaseRepresentation [5, 8] c' = BaseRepresentation [8, 5] d' →
      c + d ≤ c' + d') ∧
    c + d = 15 :=
  sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l3904_390453


namespace NUMINAMATH_CALUDE_min_value_theorem_l3904_390416

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 3/2) :
  (∀ x y : ℝ, x > 0 → y > 1 → x + y = 3/2 → 2/x + 1/(y-1) ≥ 2/a + 1/(b-1)) ∧
  2/a + 1/(b-1) = 6 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3904_390416


namespace NUMINAMATH_CALUDE_sadies_daily_burger_spending_l3904_390410

/-- Sadie's daily burger spending in June -/
def daily_burger_spending (total_spending : ℚ) (days : ℕ) : ℚ :=
  total_spending / days

theorem sadies_daily_burger_spending :
  let total_spending : ℚ := 372
  let days : ℕ := 30
  daily_burger_spending total_spending days = 12.4 := by
  sorry

end NUMINAMATH_CALUDE_sadies_daily_burger_spending_l3904_390410
