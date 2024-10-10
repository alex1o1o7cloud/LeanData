import Mathlib

namespace gcd_lcm_product_l889_88951

theorem gcd_lcm_product (a b : ℕ) (ha : a = 24) (hb : b = 54) :
  Nat.gcd a b * Nat.lcm a b = a * b := by
sorry

end gcd_lcm_product_l889_88951


namespace point_on_line_with_given_x_l889_88939

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a straight line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem point_on_line_with_given_x (l : Line) (x : ℝ) :
  l.slope = 2 →
  l.yIntercept = 2 →
  x = 269 →
  ∃ p : Point, p.x = x ∧ pointOnLine p l ∧ p.y = 540 := by
  sorry

end point_on_line_with_given_x_l889_88939


namespace students_just_passed_l889_88926

theorem students_just_passed (total_students : ℕ) 
  (first_division_percentage : ℚ) (second_division_percentage : ℚ) : 
  total_students = 300 →
  first_division_percentage = 30 / 100 →
  second_division_percentage = 54 / 100 →
  (total_students : ℚ) * (1 - first_division_percentage - second_division_percentage) = 48 := by
  sorry

end students_just_passed_l889_88926


namespace shortest_side_right_triangle_l889_88964

theorem shortest_side_right_triangle (a b c : ℝ) (ha : a = 9) (hb : b = 12) (hc : c^2 = a^2 + b^2) :
  min a (min b c) = 9 := by
  sorry

end shortest_side_right_triangle_l889_88964


namespace average_salary_l889_88915

def salary_A : ℕ := 10000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def num_individuals : ℕ := 5

theorem average_salary :
  (salary_A + salary_B + salary_C + salary_D + salary_E) / num_individuals = 8600 := by
  sorry

end average_salary_l889_88915


namespace angle_conversion_negative_1125_conversion_l889_88933

theorem angle_conversion (angle : ℝ) : ∃ (k : ℤ) (α : ℝ), 
  angle = k * 360 + α ∧ 0 ≤ α ∧ α < 360 :=
by
  -- Proof goes here
  sorry

theorem negative_1125_conversion : 
  ∃ (k : ℤ) (α : ℝ), -1125 = k * 360 + α ∧ 0 ≤ α ∧ α < 360 ∧ k = -4 ∧ α = 315 :=
by
  -- Proof goes here
  sorry

end angle_conversion_negative_1125_conversion_l889_88933


namespace store_profit_maximization_l889_88922

/-- Represents the store's profit function --/
def profit_function (x : ℕ) : ℚ := -10 * x^2 + 800 * x + 20000

/-- Represents the constraint on the price increase --/
def valid_price_increase (x : ℕ) : Prop := x ≤ 100

theorem store_profit_maximization :
  ∃ (x : ℕ), valid_price_increase x ∧
    (∀ (y : ℕ), valid_price_increase y → profit_function y ≤ profit_function x) ∧
    x = 40 ∧ profit_function x = 36000 := by
  sorry

#check store_profit_maximization

end store_profit_maximization_l889_88922


namespace smallest_distance_between_circles_l889_88984

theorem smallest_distance_between_circles (z w : ℂ) : 
  Complex.abs (z - (2 + 2 * Complex.I)) = 2 →
  Complex.abs (w - (5 + 6 * Complex.I)) = 4 →
  ∃ (min_dist : ℝ), min_dist = 11 ∧ 
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 + 2 * Complex.I)) = 2 →
                   Complex.abs (w' - (5 + 6 * Complex.I)) = 4 →
                   Complex.abs (z' - w') ≥ min_dist :=
by
  sorry

end smallest_distance_between_circles_l889_88984


namespace three_digit_square_mod_1000_l889_88987

theorem three_digit_square_mod_1000 (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999) → (n^2 ≡ n [ZMOD 1000]) ↔ (n = 376 ∨ n = 625) := by
  sorry

end three_digit_square_mod_1000_l889_88987


namespace inequality_proof_l889_88972

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b^2 / a + c^2 / b + a^2 / c ≥ a + b + c := by
  sorry

end inequality_proof_l889_88972


namespace intersection_forms_line_l889_88949

-- Define the equations
def hyperbola (x y : ℝ) : Prop := x * y = 12
def ellipse (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 36) = 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ hyperbola x y ∧ ellipse x y}

-- Theorem statement
theorem intersection_forms_line :
  ∃ (a b : ℝ), ∀ (p : ℝ × ℝ), p ∈ intersection_points → 
  (p.1 = a * p.2 + b ∨ p.2 = a * p.1 + b) :=
sorry

end intersection_forms_line_l889_88949


namespace water_volume_ratio_in_cone_l889_88962

theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height := (2 : ℝ) / 3 * h
  let water_radius := (2 : ℝ) / 3 * r
  let cone_volume := (1 : ℝ) / 3 * π * r^2 * h
  let water_volume := (1 : ℝ) / 3 * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by
sorry

end water_volume_ratio_in_cone_l889_88962


namespace cylinder_radius_ratio_l889_88932

/-- Represents a right circular cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The cost of filling a cylinder with gasoline --/
def fillCost (c : Cylinder) (fullness : ℝ) : ℝ := sorry

/-- The problem statement --/
theorem cylinder_radius_ratio 
  (V B : Cylinder) 
  (h_height : V.height = B.height / 2)
  (h_cost_B : fillCost B 0.5 = 4)
  (h_cost_V : fillCost V 1 = 16) :
  V.radius / B.radius = 2 := by 
  sorry


end cylinder_radius_ratio_l889_88932


namespace total_folded_sheets_l889_88923

theorem total_folded_sheets (initial_sheets : ℕ) (additional_sheets : ℕ) : 
  initial_sheets = 45 → additional_sheets = 18 → initial_sheets + additional_sheets = 63 := by
sorry

end total_folded_sheets_l889_88923


namespace gcd_from_lcm_and_ratio_l889_88959

theorem gcd_from_lcm_and_ratio (X Y : ℕ+) :
  Nat.lcm X Y = 180 →
  (X : ℚ) / Y = 2 / 5 →
  Nat.gcd X Y = 18 := by
sorry

end gcd_from_lcm_and_ratio_l889_88959


namespace wendy_bouquets_l889_88973

/-- Represents the number of flowers of each type -/
structure FlowerCount where
  roses : ℕ
  lilies : ℕ
  daisies : ℕ

/-- Calculates the number of complete bouquets that can be made -/
def max_bouquets (initial : FlowerCount) (wilted : FlowerCount) (bouquet : FlowerCount) : ℕ :=
  let remaining : FlowerCount := ⟨
    initial.roses - wilted.roses,
    initial.lilies - wilted.lilies,
    initial.daisies - wilted.daisies
  ⟩
  min (remaining.roses / bouquet.roses)
      (min (remaining.lilies / bouquet.lilies)
           (remaining.daisies / bouquet.daisies))

/-- The main theorem stating that the maximum number of complete bouquets is 2 -/
theorem wendy_bouquets :
  let initial : FlowerCount := ⟨20, 15, 10⟩
  let wilted : FlowerCount := ⟨12, 8, 5⟩
  let bouquet : FlowerCount := ⟨3, 2, 1⟩
  max_bouquets initial wilted bouquet = 2 := by
  sorry


end wendy_bouquets_l889_88973


namespace quadratic_second_root_l889_88982

theorem quadratic_second_root 
  (p q r : ℝ) 
  (h : 2*p*(q-r)*2^2 + 3*q*(r-p)*2 + 4*r*(p-q) = 0) :
  ∃ x : ℝ, 
    x ≠ 2 ∧ 
    2*p*(q-r)*x^2 + 3*q*(r-p)*x + 4*r*(p-q) = 0 ∧ 
    x = (r*(p-q)) / (p*(q-r)) :=
by sorry

end quadratic_second_root_l889_88982


namespace matrix_equation_solution_l889_88940

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 2; 1, 0, 1; 2, 1, 0]

theorem matrix_equation_solution :
  ∃ (p q r : ℤ), A^3 + p • A^2 + q • A + r • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 ∧ p = 0 ∧ q = -6 ∧ r = -4 := by
  sorry

end matrix_equation_solution_l889_88940


namespace brown_dogs_count_l889_88912

/-- Proves the number of brown dogs in a kennel with specific conditions -/
theorem brown_dogs_count (total : ℕ) (long_fur : ℕ) (neither : ℕ) (long_fur_brown : ℕ)
  (h1 : total = 45)
  (h2 : long_fur = 26)
  (h3 : neither = 8)
  (h4 : long_fur_brown = 19) :
  total - long_fur + long_fur_brown = 30 := by
  sorry

#check brown_dogs_count

end brown_dogs_count_l889_88912


namespace incorrect_stability_statement_l889_88999

/-- Represents the variance of an individual's high jump scores -/
structure JumpVariance where
  value : ℝ
  is_positive : value > 0

/-- Represents the stability of an individual's high jump scores -/
def more_stable (a b : JumpVariance) : Prop :=
  a.value < b.value

theorem incorrect_stability_statement :
  ∃ (a b : JumpVariance),
    a.value = 1.1 ∧
    b.value = 2.5 ∧
    ¬(more_stable a b) :=
sorry

end incorrect_stability_statement_l889_88999


namespace solve_equation_l889_88974

/-- Custom remainder operation Θ -/
def theta (m n : ℕ) : ℕ :=
  if m ≥ n then m % n else n % m

/-- Main theorem -/
theorem solve_equation :
  ∃ (A : ℕ), 0 < A ∧ A < 40 ∧ theta 20 (theta A 20) = 7 ∧ A = 33 :=
by sorry

end solve_equation_l889_88974


namespace max_value_of_five_numbers_l889_88970

theorem max_value_of_five_numbers (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- five distinct natural numbers
  (a + b + c + d + e) / 5 = 15 →  -- average is 15
  c = 18 →  -- median is 18
  e ≤ 37 := by
sorry

end max_value_of_five_numbers_l889_88970


namespace possible_values_of_a_l889_88935

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 2 ∨ a = -2 := by
  sorry

end possible_values_of_a_l889_88935


namespace tan_equality_solution_l889_88996

theorem tan_equality_solution (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1500 * π / 180) →
  n = 60 ∨ n = -120 :=
by sorry

end tan_equality_solution_l889_88996


namespace greatest_difference_of_units_digit_l889_88910

theorem greatest_difference_of_units_digit (x : ℕ) : 
  x < 10 →
  (720 + x) % 4 = 0 →
  ∃ y z, y < 10 ∧ z < 10 ∧ 
         (720 + y) % 4 = 0 ∧ 
         (720 + z) % 4 = 0 ∧ 
         y - z ≤ 8 ∧
         ∀ w, w < 10 → (720 + w) % 4 = 0 → y - w ≤ 8 ∧ w - z ≤ 8 := by
  sorry

end greatest_difference_of_units_digit_l889_88910


namespace days_worked_by_c_l889_88904

-- Define the problem parameters
def days_a : ℕ := 6
def days_b : ℕ := 9
def wage_ratio_a : ℕ := 3
def wage_ratio_b : ℕ := 4
def wage_ratio_c : ℕ := 5
def daily_wage_c : ℕ := 100
def total_earning : ℕ := 1480

-- Theorem statement
theorem days_worked_by_c :
  ∃ (days_c : ℕ),
    days_c * daily_wage_c +
    days_a * (daily_wage_c * wage_ratio_a / wage_ratio_c) +
    days_b * (daily_wage_c * wage_ratio_b / wage_ratio_c) = total_earning ∧
    days_c = 4 := by
  sorry


end days_worked_by_c_l889_88904


namespace indeterminate_f_five_l889_88981

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem indeterminate_f_five
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_shift : ∀ x, f 1 = f (x + 2) ∧ f (x + 2) = f x + f 2) :
  ¬∃ y, ∀ f, IsOdd f → (∀ x, f 1 = f (x + 2) ∧ f (x + 2) = f x + f 2) → f 5 = y :=
sorry

end indeterminate_f_five_l889_88981


namespace parallel_lines_l889_88969

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle Γ
def circle_gamma (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 8

-- Define point M on Γ
def point_on_gamma (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  circle_gamma x y

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define circle ⊙F
def circle_F (M : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  let (x_m, y_m) := M
  let (x_p, y_p) := P
  (x_p - 1)^2 + y_p^2 = (x_m - 1)^2 + y_m^2

-- Define line l tangent to ⊙F at M
def line_l (M A B : ℝ × ℝ) : Prop :=
  ∃ (k b : ℝ), ∀ (x y : ℝ),
    ((x, y) = M ∨ (x, y) = A ∨ (x, y) = B) → y = k*x + b

-- Define lines l₁ and l₂
def line_l1_l2 (A B Q R : ℝ × ℝ) : Prop :=
  ∃ (k1 b1 k2 b2 : ℝ),
    (∀ (x y : ℝ), ((x, y) = A ∨ (x, y) = Q) → y = k1*x + b1) ∧
    (∀ (x y : ℝ), ((x, y) = B ∨ (x, y) = R) → y = k2*x + b2)

-- Main theorem
theorem parallel_lines
  (M A B Q R : ℝ × ℝ)
  (h_M : point_on_gamma M)
  (h_A : parabola A.1 A.2)
  (h_B : parabola B.1 B.2)
  (h_l : line_l M A B)
  (h_F : circle_F M Q)
  (h_F' : circle_F M R)
  (h_l1_l2 : line_l1_l2 A B Q R) :
  -- Conclusion: l₁ is parallel to l₂
  ∃ (k1 b1 k2 b2 : ℝ),
    (∀ (x y : ℝ), ((x, y) = A ∨ (x, y) = Q) → y = k1*x + b1) ∧
    (∀ (x y : ℝ), ((x, y) = B ∨ (x, y) = R) → y = k2*x + b2) ∧
    k1 = k2 :=
by sorry

end parallel_lines_l889_88969


namespace caravan_keepers_count_l889_88948

/-- Represents the number of feet for different animals and humans -/
def feet_count : Nat → Nat
| 0 => 2  -- humans and hens
| 1 => 4  -- goats and camels
| _ => 0

/-- The caravan problem -/
theorem caravan_keepers_count 
  (hens goats camels : Nat) 
  (hens_count : hens = 60)
  (goats_count : goats = 35)
  (camels_count : camels = 6)
  (feet_head_diff : 
    ∃ keepers : Nat, 
      hens * feet_count 0 + 
      goats * feet_count 1 + 
      camels * feet_count 1 + 
      keepers * feet_count 0 = 
      hens + goats + camels + keepers + 193) :
  ∃ keepers : Nat, keepers = 10 := by
sorry

end caravan_keepers_count_l889_88948


namespace equation_solution_l889_88918

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 3), (2, 3, 2), (3, 2, 2), (5, 1, 4), (5, 4, 1), (4, 1, 5), (4, 5, 1),
   (1, 4, 5), (1, 5, 4), (8, 1, 3), (8, 3, 1), (3, 1, 8), (3, 8, 1), (1, 3, 8), (1, 8, 3)}

def satisfies_equation (x y z : ℕ) : Prop :=
  (x + 1) * (y + 1) * (z + 1) = 3 * x * y * z

theorem equation_solution :
  ∀ x y z : ℕ, satisfies_equation x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end equation_solution_l889_88918


namespace zero_function_solution_l889_88909

def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x - f y

theorem zero_function_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f x = 0 := by
  sorry

end zero_function_solution_l889_88909


namespace friend_has_five_balloons_l889_88975

/-- The number of balloons you have -/
def your_balloons : ℕ := 7

/-- The difference between your balloons and your friend's balloons -/
def difference : ℕ := 2

/-- The number of balloons your friend has -/
def friend_balloons : ℕ := your_balloons - difference

theorem friend_has_five_balloons : friend_balloons = 5 := by
  sorry

end friend_has_five_balloons_l889_88975


namespace max_value_sine_cosine_sum_l889_88978

theorem max_value_sine_cosine_sum :
  let f : ℝ → ℝ := λ x ↦ 6 * Real.sin x + 8 * Real.cos x
  ∃ M : ℝ, M = 10 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end max_value_sine_cosine_sum_l889_88978


namespace apple_cost_proof_l889_88993

theorem apple_cost_proof (original_price : ℝ) (price_increase : ℝ) (family_size : ℕ) (pounds_per_person : ℝ) : 
  original_price = 1.6 → 
  price_increase = 0.25 → 
  family_size = 4 → 
  pounds_per_person = 2 → 
  (original_price + original_price * price_increase) * (family_size : ℝ) * pounds_per_person = 16 := by
sorry

end apple_cost_proof_l889_88993


namespace square_perimeter_diagonal_ratio_l889_88957

theorem square_perimeter_diagonal_ratio 
  (s₁ s₂ : ℝ) 
  (h_positive₁ : s₁ > 0) 
  (h_positive₂ : s₂ > 0) 
  (h_perimeter_ratio : 4 * s₂ = 5 * (4 * s₁)) :
  s₂ * Real.sqrt 2 = 5 * (s₁ * Real.sqrt 2) := by
sorry

end square_perimeter_diagonal_ratio_l889_88957


namespace sequence_is_quadratic_l889_88905

/-- Checks if a sequence is consistent with a quadratic function --/
def is_quadratic_sequence (seq : List ℕ) : Prop :=
  let first_differences := List.zipWith (·-·) (seq.tail) seq
  let second_differences := List.zipWith (·-·) (first_differences.tail) first_differences
  second_differences.all (· = second_differences.head!)

/-- The given sequence of function values --/
def given_sequence : List ℕ := [1600, 1764, 1936, 2116, 2304, 2500, 2704, 2916]

theorem sequence_is_quadratic :
  is_quadratic_sequence given_sequence :=
sorry

end sequence_is_quadratic_l889_88905


namespace min_copies_discount_proof_l889_88994

/-- The minimum number of photocopies required for a discount -/
def min_copies_for_discount : ℕ := 160

/-- The cost of one photocopy in dollars -/
def cost_per_copy : ℚ := 2 / 100

/-- The discount rate offered -/
def discount_rate : ℚ := 25 / 100

/-- The total savings when ordering 160 copies -/
def total_savings : ℚ := 80 / 100

theorem min_copies_discount_proof :
  (min_copies_for_discount : ℚ) * cost_per_copy * (1 - discount_rate) =
  (min_copies_for_discount : ℚ) * cost_per_copy - total_savings :=
by sorry

end min_copies_discount_proof_l889_88994


namespace abc_relationship_l889_88925

-- Define the constants a, b, and c
noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log 3

-- State the theorem
theorem abc_relationship : b > c ∧ c > a := by sorry

end abc_relationship_l889_88925


namespace barbell_cost_l889_88956

/-- Given that John buys barbells, gives money, and receives change, 
    prove the cost of each barbell. -/
theorem barbell_cost (num_barbells : ℕ) (money_given : ℕ) (change_received : ℕ) : 
  num_barbells = 3 → money_given = 850 → change_received = 40 → 
  (money_given - change_received) / num_barbells = 270 := by
  sorry

#check barbell_cost

end barbell_cost_l889_88956


namespace find_M_l889_88945

theorem find_M : ∃ M : ℚ, (10 + 11 + 12) / 3 = (2022 + 2023 + 2024) / M ∧ M = 551 := by
  sorry

end find_M_l889_88945


namespace probability_a_equals_one_l889_88995

theorem probability_a_equals_one (a b c : ℕ+) (sum_constraint : a + b + c = 6) :
  (Finset.filter (fun x => x.1 = 1) (Finset.product (Finset.range 6) (Finset.product (Finset.range 6) (Finset.range 6)))).card /
  (Finset.filter (fun x => x.1 + x.2.1 + x.2.2 = 6) (Finset.product (Finset.range 6) (Finset.product (Finset.range 6) (Finset.range 6)))).card
  = 2 / 5 := by
  sorry

end probability_a_equals_one_l889_88995


namespace julia_age_after_ten_years_l889_88921

/-- Given the ages and relationships of siblings, calculate Julia's age after 10 years -/
theorem julia_age_after_ten_years 
  (justin_age : ℕ)
  (jessica_age_when_justin_born : ℕ)
  (james_age_diff_jessica : ℕ)
  (julia_age_diff_justin : ℕ)
  (h1 : justin_age = 26)
  (h2 : jessica_age_when_justin_born = 6)
  (h3 : james_age_diff_jessica = 7)
  (h4 : julia_age_diff_justin = 8) :
  justin_age - julia_age_diff_justin + 10 = 28 :=
by sorry

end julia_age_after_ten_years_l889_88921


namespace flag_count_l889_88937

/-- The number of colors available for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flags : ℕ := num_colors ^ num_stripes

theorem flag_count : total_flags = 27 := by
  sorry

end flag_count_l889_88937


namespace influenza_spread_l889_88944

theorem influenza_spread (x : ℝ) : (1 + x)^2 = 100 → x = 9 := by sorry

end influenza_spread_l889_88944


namespace largest_square_area_l889_88920

/-- Given a configuration of 7 squares where the smallest square has area 9,
    the largest square has area 324. -/
theorem largest_square_area (num_squares : ℕ) (smallest_area : ℝ) : 
  num_squares = 7 → smallest_area = 9 → ∃ (largest_area : ℝ), largest_area = 324 := by
  sorry

end largest_square_area_l889_88920


namespace boy_running_duration_l889_88919

theorem boy_running_duration (initial_speed initial_time second_distance second_speed : ℝ) 
  (h1 : initial_speed = 15)
  (h2 : initial_time = 3)
  (h3 : second_distance = 190)
  (h4 : second_speed = 19) : 
  initial_time + second_distance / second_speed = 13 := by
  sorry

end boy_running_duration_l889_88919


namespace savings_multiple_l889_88952

/-- Represents a worker's monthly finances -/
structure WorkerFinances where
  takehome : ℝ  -- Monthly take-home pay
  savingsRate : ℝ  -- Fraction of take-home pay saved each month
  months : ℕ  -- Number of months

/-- Calculates the total amount saved over a given number of months -/
def totalSaved (w : WorkerFinances) : ℝ :=
  w.takehome * w.savingsRate * w.months

/-- Calculates the amount not saved in one month -/
def monthlyUnsaved (w : WorkerFinances) : ℝ :=
  w.takehome * (1 - w.savingsRate)

/-- Theorem stating that for a worker saving 1/4 of their take-home pay,
    the total saved over 12 months is 4 times the monthly unsaved amount -/
theorem savings_multiple (w : WorkerFinances)
    (h1 : w.savingsRate = 1/4)
    (h2 : w.months = 12) :
    totalSaved w = 4 * monthlyUnsaved w := by
  sorry


end savings_multiple_l889_88952


namespace alcohol_mixture_problem_l889_88971

theorem alcohol_mixture_problem (original_volume : ℝ) (added_water : ℝ) (new_percentage : ℝ) :
  original_volume = 15 →
  added_water = 2 →
  new_percentage = 17.647058823529413 →
  (new_percentage / 100) * (original_volume + added_water) = (20 / 100) * original_volume :=
by sorry

end alcohol_mixture_problem_l889_88971


namespace jack_morning_emails_l889_88980

/-- Given that Jack received 3 emails in the afternoon, 1 email in the evening,
    and a total of 10 emails in the day, prove that he received 6 emails in the morning. -/
theorem jack_morning_emails
  (total : ℕ)
  (afternoon : ℕ)
  (evening : ℕ)
  (h1 : total = 10)
  (h2 : afternoon = 3)
  (h3 : evening = 1) :
  total - (afternoon + evening) = 6 :=
by sorry

end jack_morning_emails_l889_88980


namespace range_of_m_l889_88900

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (m - 1 < x ∧ x < m + 1) → (x^2 - 2*x - 3 > 0)) ∧ 
  (∃ x : ℝ, (x^2 - 2*x - 3 > 0) ∧ ¬(m - 1 < x ∧ x < m + 1)) ↔ 
  (m ≤ -2 ∨ m ≥ 4) :=
sorry

end range_of_m_l889_88900


namespace hypotenuse_to_brush_ratio_l889_88928

/-- A right triangle with hypotenuse 2a and a brush of width w painting one-third of its area -/
structure PaintedTriangle (a : ℝ) where
  w : ℝ
  area_painted : (a ^ 2) / 3 = a * w

/-- The ratio of the hypotenuse to the brush width is 6 -/
theorem hypotenuse_to_brush_ratio (a : ℝ) (t : PaintedTriangle a) :
  (2 * a) / t.w = 6 := by sorry

end hypotenuse_to_brush_ratio_l889_88928


namespace nail_size_fraction_l889_88979

theorem nail_size_fraction (x : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 1) 
  (h2 : x + 0.5 = 0.75) : 
  x = 0.25 := by
sorry

end nail_size_fraction_l889_88979


namespace units_digit_of_expression_l889_88991

theorem units_digit_of_expression : 
  (30 * 32 * 34 * 36 * 38 * 40) / 2000 ≡ 6 [ZMOD 10] := by
  sorry

end units_digit_of_expression_l889_88991


namespace base_of_first_term_l889_88924

/-- Given a positive integer h that is divisible by both 225 and 216,
    and can be expressed as h = x^a * 3^b * 5^c where x, a, b, and c are positive integers,
    and the least possible value of a + b + c is 8,
    prove that x must be 2. -/
theorem base_of_first_term (h : ℕ+) (x : ℕ+) (a b c : ℕ+) 
  (h_div_225 : 225 ∣ h)
  (h_div_216 : 216 ∣ h)
  (h_eq : h = x^(a:ℕ) * 3^(b:ℕ) * 5^(c:ℕ))
  (abc_min : a + b + c = 8 ∧ ∀ a' b' c' : ℕ+, a' + b' + c' ≥ 8) : x = 2 :=
sorry

end base_of_first_term_l889_88924


namespace tv_cost_is_1060_l889_88946

-- Define the given values
def total_initial_purchase : ℝ := 3000
def returned_bike_cost : ℝ := 500
def toaster_cost : ℝ := 100
def total_out_of_pocket : ℝ := 2020

-- Define the TV cost as a variable
def tv_cost : ℝ := sorry

-- Define the sold bike cost
def sold_bike_cost : ℝ := returned_bike_cost * 1.2

-- Define the sale price of the sold bike
def sold_bike_sale_price : ℝ := sold_bike_cost * 0.8

-- Theorem stating that the TV cost is $1060
theorem tv_cost_is_1060 :
  tv_cost = 1060 :=
by
  sorry

#check tv_cost_is_1060

end tv_cost_is_1060_l889_88946


namespace code_deciphering_probability_l889_88934

theorem code_deciphering_probability 
  (p_a p_b : ℝ) 
  (h_a : p_a = 0.3) 
  (h_b : p_b = 0.3) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - p_a) * (1 - p_b) = 0.51 :=
sorry

end code_deciphering_probability_l889_88934


namespace survey_total_is_120_l889_88990

/-- Represents the survey results of parents' ratings on their children's online class experience -/
structure SurveyResults where
  total : ℕ
  excellent : ℕ
  verySatisfactory : ℕ
  satisfactory : ℕ
  needsImprovement : ℕ

/-- The conditions of the survey results -/
def surveyConditions (s : SurveyResults) : Prop :=
  s.excellent = (15 * s.total) / 100 ∧
  s.verySatisfactory = (60 * s.total) / 100 ∧
  s.satisfactory = (80 * (s.total - s.excellent - s.verySatisfactory)) / 100 ∧
  s.needsImprovement = s.total - s.excellent - s.verySatisfactory - s.satisfactory ∧
  s.needsImprovement = 6

/-- Theorem stating that the total number of parents who answered the survey is 120 -/
theorem survey_total_is_120 (s : SurveyResults) (h : surveyConditions s) : s.total = 120 := by
  sorry

end survey_total_is_120_l889_88990


namespace dot_product_AB_normal_is_zero_l889_88929

def A : ℝ × ℝ := (3, -1)
def B : ℝ × ℝ := (6, 1)
def l (x y : ℝ) : Prop := 2 * x - 3 * y - 9 = 0

def normal_vector (l : (ℝ → ℝ → Prop)) : ℝ × ℝ := (2, -3)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem dot_product_AB_normal_is_zero :
  (vector_AB.1 * (normal_vector l).1 + vector_AB.2 * (normal_vector l).2) = 0 := by
  sorry

end dot_product_AB_normal_is_zero_l889_88929


namespace inverse_abs_equality_false_l889_88903

theorem inverse_abs_equality_false : ¬ ∀ a b : ℝ, |a| = |b| → a = b := by
  sorry

end inverse_abs_equality_false_l889_88903


namespace smallest_n_complex_equality_l889_88967

theorem smallest_n_complex_equality (a b : ℝ) (c : ℕ+) 
  (h_a : a > 0) (h_b : b > 0) 
  (h_smallest : ∀ k : ℕ+, k < 3 → (c * a + b * Complex.I) ^ k.val ≠ (c * a - b * Complex.I) ^ k.val) 
  (h_equal : (c * a + b * Complex.I) ^ 3 = (c * a - b * Complex.I) ^ 3) :
  b / (c * a) = Real.sqrt 3 := by
sorry

end smallest_n_complex_equality_l889_88967


namespace tangent_line_and_max_value_l889_88938

-- Define the function f(x) = x³ - ax²
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem tangent_line_and_max_value (a : ℝ) :
  f' a 1 = 3 →
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x + x ≥ 0) →
  (∃ (m b : ℝ), m = 3 ∧ b = -2 ∧
    ∀ x : ℝ, f a x = m * (x - 1) + f a 1) ∧
  (∃ M : ℝ, M = 8 ∧
    ∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x ≤ M) ∧
  a ≤ 2 :=
by sorry


end tangent_line_and_max_value_l889_88938


namespace chocolate_bars_original_count_l889_88942

/-- The number of chocolate bars remaining after eating a certain percentage each day for a given number of days -/
def remaining_bars (initial : ℕ) (eat_percentage : ℚ) (days : ℕ) : ℚ :=
  initial * (1 - eat_percentage) ^ days

/-- The theorem stating the original number of chocolate bars given the remaining bars after 4 days -/
theorem chocolate_bars_original_count :
  ∃ (initial : ℕ),
    remaining_bars initial (30 / 100) 4 = 16 ∧
    initial = 67 :=
sorry

end chocolate_bars_original_count_l889_88942


namespace polynomial_factorization_l889_88966

theorem polynomial_factorization (x : ℝ) : 
  x^12 + x^6 + 1 = (x^2 + 1) * (x^4 - x^2 + 1)^2 := by
  sorry

end polynomial_factorization_l889_88966


namespace symmetric_line_wrt_y_axis_l889_88917

/-- Given a line with equation x - 2y + 1 = 0, its symmetric line with respect to the y-axis
    has the equation x + 2y - 1 = 0 -/
theorem symmetric_line_wrt_y_axis :
  ∀ (x y : ℝ), x - 2*y + 1 = 0 → ∃ (x' y' : ℝ), x' + 2*y' - 1 = 0 ∧ x' = -x ∧ y' = y :=
sorry

end symmetric_line_wrt_y_axis_l889_88917


namespace solve_grocery_problem_l889_88963

def grocery_problem (total_brought chicken veggies eggs dog_food left_after meat : ℕ) : Prop :=
  total_brought = 167 ∧
  chicken = 22 ∧
  veggies = 43 ∧
  eggs = 5 ∧
  dog_food = 45 ∧
  left_after = 35 ∧
  meat = total_brought - (chicken + veggies + eggs + dog_food + left_after)

theorem solve_grocery_problem :
  ∃ meat, grocery_problem 167 22 43 5 45 35 meat ∧ meat = 17 := by sorry

end solve_grocery_problem_l889_88963


namespace binomial_12_choose_10_l889_88998

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_choose_10_l889_88998


namespace fish_count_l889_88960

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 14

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 24 := by
  sorry

end fish_count_l889_88960


namespace simplify_expression_l889_88997

theorem simplify_expression (r : ℝ) : 120*r - 38*r + 25*r = 107*r := by
  sorry

end simplify_expression_l889_88997


namespace flight_passengers_l889_88913

theorem flight_passengers :
  ∀ (total_passengers : ℕ),
    (total_passengers : ℝ) * 0.4 = total_passengers * 0.4 →
    (total_passengers : ℝ) * 0.1 = total_passengers * 0.1 →
    (total_passengers : ℝ) * 0.9 = total_passengers - total_passengers * 0.1 →
    (total_passengers * 0.1 : ℝ) * (2/3) = total_passengers * 0.1 * (2/3) →
    (total_passengers : ℝ) * 0.4 - total_passengers * 0.1 * (2/3) = 40 →
    total_passengers = 120 :=
by sorry

end flight_passengers_l889_88913


namespace fish_pond_problem_l889_88907

theorem fish_pond_problem (initial_fish : ℕ) : 
  (∃ (initial_tadpoles : ℕ),
    initial_tadpoles = 3 * initial_fish ∧
    initial_tadpoles / 2 = (initial_fish - 7) + 32) →
  initial_fish = 50 := by
sorry

end fish_pond_problem_l889_88907


namespace min_value_theorem_l889_88992

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.sqrt (b / a) + Real.sqrt (a / b) - 2 = (Real.sqrt (a * b) - 4 * a * b) / (2 * a * b)) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 2 + 6 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → 
  Real.sqrt (y / x) + Real.sqrt (x / y) - 2 = (Real.sqrt (x * y) - 4 * x * y) / (2 * x * y) →
  1 / x + 2 / y ≥ min := by
sorry

end min_value_theorem_l889_88992


namespace greatest_value_quadratic_inequality_l889_88902

theorem greatest_value_quadratic_inequality :
  ∀ a : ℝ, -a^2 + 9*a - 20 ≥ 0 → a ≤ 5 :=
by sorry

end greatest_value_quadratic_inequality_l889_88902


namespace toy_shopping_total_l889_88950

def calculate_total_spent (prices : List Float) (discount_rate : Float) (tax_rate : Float) : Float :=
  let total_before_discount := prices.sum
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let sales_tax := tax_rate * total_after_discount
  total_after_discount + sales_tax

theorem toy_shopping_total (prices : List Float) 
  (h1 : prices = [8.25, 6.59, 12.10, 15.29, 23.47])
  (h2 : calculate_total_spent prices 0.10 0.06 = 62.68) : 
  calculate_total_spent prices 0.10 0.06 = 62.68 := by
  sorry

end toy_shopping_total_l889_88950


namespace age_difference_l889_88954

/-- The age difference between A and C, given the condition about total ages -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 11) : A = C + 11 := by
  sorry

end age_difference_l889_88954


namespace blueprint_to_actual_length_l889_88977

/-- Given a blueprint scale and a length on the blueprint, calculates the actual length in meters. -/
def actual_length (scale : ℚ) (blueprint_length : ℚ) : ℚ :=
  blueprint_length * scale / 100

/-- Proves that for a blueprint scale of 1:50 and a line segment of 10 cm on the blueprint,
    the actual length is 5 m. -/
theorem blueprint_to_actual_length :
  let scale : ℚ := 50
  let blueprint_length : ℚ := 10
  actual_length scale blueprint_length = 5 := by
  sorry

end blueprint_to_actual_length_l889_88977


namespace exists_scientist_with_one_friend_l889_88936

-- Define the type for scientists
variable (Scientist : Type)

-- Define the friendship relation
variable (is_friend : Scientist → Scientist → Prop)

-- Define the number of friends function
variable (num_friends : Scientist → ℕ)

-- State the theorem
theorem exists_scientist_with_one_friend
  (h1 : ∀ (s1 s2 : Scientist), num_friends s1 = num_friends s2 → ¬∃ (s3 : Scientist), is_friend s1 s3 ∧ is_friend s2 s3)
  (h2 : ∀ (s1 s2 : Scientist), is_friend s1 s2 → is_friend s2 s1)
  (h3 : ∀ (s : Scientist), ¬is_friend s s)
  : ∃ (s : Scientist), num_friends s = 1 :=
by sorry

end exists_scientist_with_one_friend_l889_88936


namespace point_on_circle_x_value_l889_88941

-- Define the circle
def circle_center : ℝ × ℝ := (12, 0)
def circle_radius : ℝ := 15

-- Define the point on the circle
def point_on_circle (x : ℝ) : ℝ × ℝ := (x, 12)

-- Theorem statement
theorem point_on_circle_x_value (x : ℝ) :
  (point_on_circle x).1 - circle_center.1 ^ 2 + 
  (point_on_circle x).2 - circle_center.2 ^ 2 = circle_radius ^ 2 →
  x = 3 ∨ x = 21 := by
  sorry

end point_on_circle_x_value_l889_88941


namespace banana_price_reduction_l889_88911

/-- Proves that a price reduction resulting in 64 more bananas for Rs. 40.00001 
    and a new price of Rs. 3 per dozen represents a 40% reduction from the original price. -/
theorem banana_price_reduction (original_price : ℚ) : 
  (40.00001 / 3 - 40.00001 / original_price = 64 / 12) →
  (3 / original_price = 0.6) := by
  sorry

#eval (1 - 3/5) * 100 -- Should evaluate to 40

end banana_price_reduction_l889_88911


namespace imaginary_part_of_i_times_i_plus_two_l889_88968

theorem imaginary_part_of_i_times_i_plus_two (i : ℂ) : 
  Complex.im (i * (i + 2)) = 2 :=
by sorry

end imaginary_part_of_i_times_i_plus_two_l889_88968


namespace find_number_l889_88986

theorem find_number : ∃! x : ℚ, (172 / 4 - 28) * x + 7 = 172 := by sorry

end find_number_l889_88986


namespace triangle_expression_simplification_l889_88955

theorem triangle_expression_simplification
  (a b c : ℝ)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  |a + c - b| - |a + b + c| + |2*b + c| = c :=
sorry

end triangle_expression_simplification_l889_88955


namespace product_of_repeating_decimals_l889_88931

/-- The product of two specific repeating decimals -/
theorem product_of_repeating_decimals :
  (63 : ℚ) / 99 * (54 : ℚ) / 99 = (14 : ℚ) / 41 := by
  sorry

#check product_of_repeating_decimals

end product_of_repeating_decimals_l889_88931


namespace sum_value_l889_88985

def S : ℚ := 3003 + (1/3) * (3002 + (1/6) * (3001 + (1/9) * (3000 + (1/(3*1000)) * 3)))

theorem sum_value : S = 3002.5 := by sorry

end sum_value_l889_88985


namespace shifted_parabola_equation_l889_88914

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the vertical shift
def vertical_shift : ℝ := 5

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := original_parabola x + vertical_shift

-- Theorem stating that the shifted parabola is equivalent to y = x^2 + 5
theorem shifted_parabola_equation :
  ∀ x : ℝ, shifted_parabola x = x^2 + 5 := by sorry

end shifted_parabola_equation_l889_88914


namespace angle_from_coordinates_l889_88916

theorem angle_from_coordinates (a : Real) (h1 : 0 < a) (h2 : a < π / 2) :
  ∃ (P : ℝ × ℝ), P.1 = 4 * Real.sin 3 ∧ P.2 = -4 * Real.cos 3 →
  a = 3 - π / 2 :=
sorry

end angle_from_coordinates_l889_88916


namespace sugar_per_chocolate_bar_l889_88953

/-- Given a company that produces chocolate bars, this theorem proves
    the amount of sugar needed per bar based on production rate and sugar usage. -/
theorem sugar_per_chocolate_bar
  (bars_per_minute : ℕ)
  (sugar_per_two_minutes : ℕ)
  (h1 : bars_per_minute = 36)
  (h2 : sugar_per_two_minutes = 108) :
  (sugar_per_two_minutes : ℚ) / ((bars_per_minute : ℚ) * 2) = 3/2 := by
  sorry

end sugar_per_chocolate_bar_l889_88953


namespace no_equilateral_grid_triangle_l889_88965

/-- A point with integer coordinates -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three grid points -/
structure GridTriangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint

/-- Check if a triangle is equilateral -/
def isEquilateral (t : GridTriangle) : Prop :=
  let d1 := (t.a.x - t.b.x)^2 + (t.a.y - t.b.y)^2
  let d2 := (t.b.x - t.c.x)^2 + (t.b.y - t.c.y)^2
  let d3 := (t.c.x - t.a.x)^2 + (t.c.y - t.a.y)^2
  d1 = d2 ∧ d2 = d3

/-- The main theorem: no equilateral triangle exists on the integer grid -/
theorem no_equilateral_grid_triangle :
  ¬ ∃ t : GridTriangle, isEquilateral t :=
sorry

end no_equilateral_grid_triangle_l889_88965


namespace consecutive_integers_operation_l889_88961

theorem consecutive_integers_operation (n : ℕ) (h1 : n = 9) : 
  let f : ℕ → ℕ → ℕ := λ x y => x + y + 162
  f n (n + 1) = n * (n + 1) + 91 := by
  sorry

end consecutive_integers_operation_l889_88961


namespace smallest_right_triangle_area_l889_88906

theorem smallest_right_triangle_area :
  let side1 : ℝ := 6
  let side2 : ℝ := 8
  let area1 : ℝ := (1/2) * side1 * side2
  let area2 : ℝ := (1/2) * side1 * Real.sqrt (side2^2 - side1^2)
  min area1 area2 = 6 * Real.sqrt 7 := by
  sorry

end smallest_right_triangle_area_l889_88906


namespace largest_angle_in_triangle_l889_88901

theorem largest_angle_in_triangle (a b y : ℝ) : 
  a = 60 ∧ b = 70 ∧ a + b + y = 180 → 
  max a (max b y) = 70 := by
  sorry

end largest_angle_in_triangle_l889_88901


namespace natural_fraction_condition_l889_88908

theorem natural_fraction_condition (k n : ℕ) :
  (∃ m : ℕ, (7 * k + 15 * n - 1) = m * (3 * k + 4 * n)) ↔
  (∃ a : ℕ, k = 3 * a - 2 ∧ n = 2 * a - 1) :=
by sorry

end natural_fraction_condition_l889_88908


namespace composite_function_sum_l889_88976

/-- Given a function f(x) = px + q where p and q are real numbers,
    if f(f(f(x))) = 8x + 21, then p + q = 5 -/
theorem composite_function_sum (p q : ℝ) :
  (∀ x, ∃ f : ℝ → ℝ, f x = p * x + q) →
  (∀ x, ∃ f : ℝ → ℝ, f (f (f x)) = 8 * x + 21) →
  p + q = 5 := by
  sorry

end composite_function_sum_l889_88976


namespace prob_no_increasing_pie_is_correct_l889_88943

/-- Represents the number of pies Alice has initially -/
def total_pies : ℕ := 6

/-- Represents the number of pies that increase in size -/
def increasing_pies : ℕ := 2

/-- Represents the number of pies that decrease in size -/
def decreasing_pies : ℕ := 4

/-- Represents the number of pies Alice gives to Mary -/
def pies_given : ℕ := 3

/-- Calculates the probability that one of the girls does not have a single size-increasing pie -/
def prob_no_increasing_pie : ℚ := 7/10

/-- Theorem stating that the probability of one girl having no increasing pie is 0.7 -/
theorem prob_no_increasing_pie_is_correct : 
  prob_no_increasing_pie = 7/10 :=
sorry

end prob_no_increasing_pie_is_correct_l889_88943


namespace dot_product_of_vectors_l889_88958

theorem dot_product_of_vectors (a b : ℝ × ℝ) :
  a = (2, 1) →
  ‖b‖ = Real.sqrt 3 →
  ‖a + b‖ = 4 →
  a • b = 4 := by sorry

end dot_product_of_vectors_l889_88958


namespace sum_difference_l889_88947

def mena_sequence : List Nat := List.range 30

def emily_sequence : List Nat :=
  mena_sequence.map (fun n => 
    let tens := n / 10
    let ones := n % 10
    if tens = 2 then 10 + ones
    else if ones = 2 then tens * 10 + 1
    else n)

theorem sum_difference : 
  mena_sequence.sum - emily_sequence.sum = 103 := by
  sorry

end sum_difference_l889_88947


namespace poojas_speed_l889_88930

/-- 
Given:
- Roja moves in the opposite direction from Pooja at 5 km/hr
- After 4 hours, the distance between Roja and Pooja is 32 km

Prove that Pooja's speed is 3 km/hr
-/
theorem poojas_speed (roja_speed : ℝ) (time : ℝ) (distance : ℝ) :
  roja_speed = 5 →
  time = 4 →
  distance = 32 →
  ∃ (pooja_speed : ℝ), pooja_speed = 3 ∧ distance = (roja_speed + pooja_speed) * time :=
by sorry

end poojas_speed_l889_88930


namespace binomial_coefficient_sum_equality_l889_88983

theorem binomial_coefficient_sum_equality (n : ℕ) : 4^n = 2^10 → n = 5 := by
  sorry

end binomial_coefficient_sum_equality_l889_88983


namespace root_sum_product_theorem_l889_88927

theorem root_sum_product_theorem (m : ℚ) :
  (∃ x y : ℚ, 
    (2*(x-1)*(x-3*m) = x*(m-4)) ∧ 
    (x + y = x * y) ∧
    (∀ z : ℚ, 2*z^2 + (5*m + 6)*z + 6*m = 0 ↔ (z = x ∨ z = y))) →
  m = -2/3 := by
sorry

end root_sum_product_theorem_l889_88927


namespace meal_cost_l889_88989

theorem meal_cost (total_cost : ℕ) (num_meals : ℕ) (h1 : total_cost = 21) (h2 : num_meals = 3) :
  total_cost / num_meals = 7 := by
sorry

end meal_cost_l889_88989


namespace spinster_count_spinster_count_proof_l889_88988

theorem spinster_count : ℕ → ℕ → Prop :=
  fun spinsters cats =>
    (spinsters : ℚ) / (cats : ℚ) = 2 / 7 ∧
    cats = spinsters + 55 →
    spinsters = 22

-- The proof is omitted
theorem spinster_count_proof : spinster_count 22 77 := by sorry

end spinster_count_spinster_count_proof_l889_88988
