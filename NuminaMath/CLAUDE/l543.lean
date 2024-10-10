import Mathlib

namespace function_value_difference_bound_l543_54378

theorem function_value_difference_bound
  (f : Set.Icc 0 1 → ℝ)
  (h₁ : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h₂ : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₂ - f x₁| < |x₂.val - x₁.val|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₂ - f x₁| < (1/2 : ℝ) :=
by sorry

end function_value_difference_bound_l543_54378


namespace quadratic_always_positive_l543_54333

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + 2*x + 3 > 0 := by sorry

end quadratic_always_positive_l543_54333


namespace safe_descent_possible_l543_54352

/-- Represents the cliff and rope setup --/
structure CliffSetup where
  cliff_height : ℝ
  rope_length : ℝ
  branch_height : ℝ

/-- Defines a safe descent --/
def safe_descent (setup : CliffSetup) : Prop :=
  setup.cliff_height > setup.rope_length ∧
  setup.branch_height < setup.cliff_height ∧
  setup.branch_height > 0 ∧
  setup.rope_length ≥ setup.cliff_height - setup.branch_height + setup.branch_height / 2

/-- Theorem stating that a safe descent is possible given the specific measurements --/
theorem safe_descent_possible : 
  ∃ (setup : CliffSetup), 
    setup.cliff_height = 100 ∧ 
    setup.rope_length = 75 ∧ 
    setup.branch_height = 50 ∧ 
    safe_descent setup := by
  sorry


end safe_descent_possible_l543_54352


namespace intersection_of_A_and_B_l543_54382

-- Define the sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by sorry

end intersection_of_A_and_B_l543_54382


namespace stacy_brother_growth_l543_54307

/-- Proves that Stacy's brother grew 1 inch last year -/
theorem stacy_brother_growth (stacy_initial_height stacy_final_height stacy_growth_difference : ℕ) 
  (h1 : stacy_initial_height = 50)
  (h2 : stacy_final_height = 57)
  (h3 : stacy_growth_difference = 6) :
  stacy_final_height - stacy_initial_height - stacy_growth_difference = 1 := by
  sorry

end stacy_brother_growth_l543_54307


namespace min_values_xy_and_x_plus_y_l543_54319

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : 2*x + 8*y - x*y = 0) : 
  x*y ≥ 64 ∧ x + y ≥ 18 := by sorry

end min_values_xy_and_x_plus_y_l543_54319


namespace find_M_l543_54359

theorem find_M : ∃ (M : ℕ+), (12^2 * 45^2 : ℕ) = 15^2 * M^2 ∧ M = 36 := by
  sorry

end find_M_l543_54359


namespace bottom_sphere_radius_l543_54326

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Represents three stacked spheres in a cone -/
structure StackedSpheres where
  cone : Cone
  bottomSphere : Sphere
  middleSphere : Sphere
  topSphere : Sphere

/-- The condition for the spheres to fit in the cone -/
def spheresFitInCone (s : StackedSpheres) : Prop :=
  s.bottomSphere.radius + s.middleSphere.radius + s.topSphere.radius ≤ s.cone.height

/-- The theorem stating the radius of the bottom sphere -/
theorem bottom_sphere_radius (s : StackedSpheres) 
  (h1 : s.cone.baseRadius = 8)
  (h2 : s.cone.height = 18)
  (h3 : s.middleSphere.radius = 2 * s.bottomSphere.radius)
  (h4 : s.topSphere.radius = 3 * s.bottomSphere.radius)
  (h5 : spheresFitInCone s) :
  s.bottomSphere.radius = 3 := by
  sorry

end bottom_sphere_radius_l543_54326


namespace intersection_of_A_and_B_l543_54355

def A : Set ℕ := {70, 1946, 1997, 2003}
def B : Set ℕ := {1, 10, 70, 2016}

theorem intersection_of_A_and_B : A ∩ B = {70} := by sorry

end intersection_of_A_and_B_l543_54355


namespace assignment_methods_count_l543_54314

def number_of_teachers : ℕ := 5
def number_of_question_types : ℕ := 3

/- Define a function that calculates the number of ways to assign teachers to question types -/
def assignment_methods : ℕ := sorry

/- Theorem stating that the number of assignment methods is 150 -/
theorem assignment_methods_count : assignment_methods = 150 := by sorry

end assignment_methods_count_l543_54314


namespace jack_classics_books_l543_54345

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := 6

/-- The number of books per author -/
def books_per_author : ℕ := 33

/-- Theorem: The total number of books in Jack's classics section is 198 -/
theorem jack_classics_books : num_authors * books_per_author = 198 := by
  sorry

end jack_classics_books_l543_54345


namespace base_prime_repr_450_l543_54349

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- The base prime representation of 450 is [1, 2, 2] -/
theorem base_prime_repr_450 : base_prime_repr 450 = [1, 2, 2] := by
  sorry

end base_prime_repr_450_l543_54349


namespace bill_selling_price_l543_54387

theorem bill_selling_price (original_purchase_price : ℝ) : 
  let original_selling_price := 1.1 * original_purchase_price
  let new_selling_price := 1.17 * original_purchase_price
  new_selling_price = original_selling_price + 28 →
  original_selling_price = 440 := by
sorry

end bill_selling_price_l543_54387


namespace contingency_table_confidence_level_l543_54336

/-- Represents a 2x2 contingency table -/
structure ContingencyTable :=
  (data : Matrix (Fin 2) (Fin 2) ℕ)

/-- Calculates the k^2 value for a contingency table -/
def calculate_k_squared (table : ContingencyTable) : ℝ :=
  sorry

/-- Determines the confidence level based on the k^2 value -/
def confidence_level (k_squared : ℝ) : ℝ :=
  sorry

theorem contingency_table_confidence_level :
  ∀ (table : ContingencyTable),
  calculate_k_squared table = 4.013 →
  confidence_level (calculate_k_squared table) = 0.99 :=
sorry

end contingency_table_confidence_level_l543_54336


namespace functional_equation_solution_l543_54329

/-- The functional equation problem -/
theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y + f x) = x * f y + 2) : 
  ∀ (x : ℝ), x > 0 → f x = x + 1 := by
sorry

end functional_equation_solution_l543_54329


namespace min_disks_for_vincent_l543_54337

/-- Represents the number of disks required to store files -/
def MinDisks (total_files : ℕ) (disk_capacity : ℚ) 
  (files_09 : ℕ) (files_075 : ℕ) (files_05 : ℕ) : ℕ :=
  sorry

theorem min_disks_for_vincent : 
  MinDisks 40 2 5 15 20 = 18 := by sorry

end min_disks_for_vincent_l543_54337


namespace arithmetic_mean_characterization_l543_54338

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- φ(n) is Euler's totient function -/
def phi (n : ℕ+) : ℕ := sorry

/-- One of n, τ(n), or φ(n) is the arithmetic mean of the other two -/
def is_arithmetic_mean (n : ℕ+) : Prop :=
  (n : ℚ) = (tau n + phi n) / 2 ∨
  (tau n : ℚ) = (n + phi n) / 2 ∨
  (phi n : ℚ) = (n + tau n) / 2

theorem arithmetic_mean_characterization (n : ℕ+) :
  is_arithmetic_mean n ↔ n ∈ ({1, 4, 6, 9} : Set ℕ+) := by sorry

end arithmetic_mean_characterization_l543_54338


namespace p_and_q_true_l543_54300

theorem p_and_q_true (P Q : Prop) (h : ¬(P ∧ Q) = False) : P ∧ Q :=
sorry

end p_and_q_true_l543_54300


namespace bird_flight_problem_l543_54389

theorem bird_flight_problem (h₁ h₂ w : ℝ) (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (w_pos : w > 0)
  (h₁_val : h₁ = 20) (h₂_val : h₂ = 30) (w_val : w = 50) :
  ∃ (d x : ℝ),
    d = 10 * Real.sqrt 13 ∧
    x = 20 ∧
    d = Real.sqrt (x^2 + h₂^2) ∧
    d = Real.sqrt ((w - x)^2 + h₁^2) := by
  sorry

#check bird_flight_problem

end bird_flight_problem_l543_54389


namespace point_on_terminal_side_l543_54367

/-- Given a point P (x, 3) on the terminal side of angle θ where cos θ = -4/5, prove that x = -4 -/
theorem point_on_terminal_side (x : ℝ) (θ : ℝ) : 
  (∃ P : ℝ × ℝ, P = (x, 3) ∧ P.1 = x * Real.cos θ ∧ P.2 = x * Real.sin θ) → 
  Real.cos θ = -4/5 → 
  x = -4 := by
sorry

end point_on_terminal_side_l543_54367


namespace laundry_problem_solution_l543_54395

/-- Represents the laundromat problem setup -/
structure LaundryProblem where
  washer_cost : ℚ  -- Cost per washer load in dollars
  dryer_cost : ℚ   -- Cost per 10 minutes of dryer use in dollars
  wash_loads : ℕ   -- Number of wash loads
  num_dryers : ℕ   -- Number of dryers used
  total_spent : ℚ  -- Total amount spent in dollars

/-- Calculates the time each dryer ran in minutes -/
def dryer_time (p : LaundryProblem) : ℚ :=
  let washing_cost := p.washer_cost * p.wash_loads
  let drying_cost := p.total_spent - washing_cost
  let total_drying_time := (drying_cost / p.dryer_cost) * 10
  total_drying_time / p.num_dryers

/-- Theorem stating that for the given problem setup, each dryer ran for 40 minutes -/
theorem laundry_problem_solution (p : LaundryProblem) 
  (h1 : p.washer_cost = 4)
  (h2 : p.dryer_cost = 1/4)
  (h3 : p.wash_loads = 2)
  (h4 : p.num_dryers = 3)
  (h5 : p.total_spent = 11) :
  dryer_time p = 40 := by
  sorry


end laundry_problem_solution_l543_54395


namespace abc_product_l543_54305

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 156)
  (h2 : b * (c + a) = 168)
  (h3 : c * (a + b) = 180) :
  a * b * c = 762 := by
sorry

end abc_product_l543_54305


namespace experiment_sequences_l543_54318

/-- The number of procedures in the experiment -/
def total_procedures : ℕ := 6

/-- The number of ways to place procedure A (first or last) -/
def a_placements : ℕ := 2

/-- The number of distinct units to arrange (including BC as one unit) -/
def distinct_units : ℕ := 4

/-- The number of ways to arrange B and C within their unit -/
def bc_arrangements : ℕ := 2

/-- The total number of possible sequences for the experiment procedures -/
def total_sequences : ℕ := a_placements * (distinct_units.factorial) * bc_arrangements

theorem experiment_sequences :
  total_sequences = 96 :=
sorry

end experiment_sequences_l543_54318


namespace digit_150_of_3_over_11_l543_54328

theorem digit_150_of_3_over_11 : ∃ (d : ℕ), d = 7 ∧ 
  (∀ (n : ℕ), n ≥ 1 → n ≤ 150 → 
    (3 * 10^n) % 11 = (d * 10^(150 - n)) % 11) := by
  sorry

end digit_150_of_3_over_11_l543_54328


namespace equation_solution_l543_54331

theorem equation_solution : ∃ x : ℚ, (1 / 5 + 5 / x = 12 / x + 1 / 12) ∧ x = 60 := by
  sorry

end equation_solution_l543_54331


namespace subtracted_amount_l543_54310

theorem subtracted_amount (N : ℝ) (A : ℝ) (h1 : N = 100) (h2 : 0.8 * N - A = 60) : A = 20 := by
  sorry

end subtracted_amount_l543_54310


namespace gum_distribution_l543_54396

theorem gum_distribution (john_gum cole_gum aubrey_gum : ℕ) 
  (h1 : john_gum = 54)
  (h2 : cole_gum = 45)
  (h3 : aubrey_gum = 0)
  (num_people : ℕ)
  (h4 : num_people = 3) :
  (john_gum + cole_gum + aubrey_gum) / num_people = 33 := by
  sorry

end gum_distribution_l543_54396


namespace visitors_in_scientific_notation_l543_54364

-- Define the number of visitors
def visitors : ℕ := 203000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.03 * (10 ^ 5)

-- Theorem statement
theorem visitors_in_scientific_notation :
  (visitors : ℝ) = scientific_notation := by sorry

end visitors_in_scientific_notation_l543_54364


namespace width_of_identical_rectangles_l543_54370

/-- Given six identical rectangles forming a larger rectangle PQRS, prove that the width of each identical rectangle is 30 -/
theorem width_of_identical_rectangles (w : ℝ) : 
  (6 : ℝ) * w^2 = 5400 ∧ 3 * w = 2 * (2 * w) → w = 30 := by
  sorry

end width_of_identical_rectangles_l543_54370


namespace root_exists_in_interval_l543_54360

theorem root_exists_in_interval : ∃! x : ℝ, 1/2 < x ∧ x < 1 ∧ Real.exp x = 1/x := by
  sorry

end root_exists_in_interval_l543_54360


namespace room_length_calculation_l543_54339

/-- Given a room with specified width, total paving cost, and paving rate per square meter,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 3.75 →
  total_cost = 16500 →
  rate_per_sqm = 800 →
  (total_cost / rate_per_sqm) / width = 5.5 :=
by sorry

end room_length_calculation_l543_54339


namespace cosine_in_triangle_l543_54320

/-- Given a triangle ABC with sides a and b, prove that if a = 4, b = 5, 
    and cos(B-A) = 31/32, then cos B = 9/16 -/
theorem cosine_in_triangle (A B C : ℝ) (a b c : ℝ) : 
  a = 4 → b = 5 → Real.cos (B - A) = 31/32 → Real.cos B = 9/16 := by sorry

end cosine_in_triangle_l543_54320


namespace right_triangle_sets_l543_54304

/-- A function to check if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (2, 5, 6) cannot form a right triangle --/
theorem right_triangle_sets :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 2 2 (2 * Real.sqrt 2) ∧
  ¬is_right_triangle 2 5 6 ∧
  is_right_triangle 5 12 13 :=
by sorry

end right_triangle_sets_l543_54304


namespace sum_of_k_values_l543_54342

theorem sum_of_k_values (a b c k : ℂ) : 
  a ≠ b ∧ b ≠ c ∧ c ≠ a →
  (a + 1) / (2 - b) = k ∧
  (b + 1) / (2 - c) = k ∧
  (c + 1) / (2 - a) = k →
  ∃ k₁ k₂ : ℂ, k = k₁ ∨ k = k₂ ∧ k₁ + k₂ = (3/2 : ℂ) :=
by sorry

end sum_of_k_values_l543_54342


namespace quadratic_root_condition_l543_54361

theorem quadratic_root_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*a*x + a + 1 = 0 ∧ y^2 + 2*a*y + a + 1 = 0 ∧ x > 2 ∧ y < 2) → 
  a < -1 := by
sorry

end quadratic_root_condition_l543_54361


namespace inequality_proof_l543_54394

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end inequality_proof_l543_54394


namespace length_of_AB_l543_54376

-- Define the curves (M) and (N)
def curve_M (x y : ℝ) : Prop := x - y = 1

def curve_N (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_M A.1 A.2 ∧ curve_N A.1 A.2 ∧
  curve_M B.1 B.2 ∧ curve_N B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 :=
sorry

end length_of_AB_l543_54376


namespace students_playing_neither_l543_54379

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 38 →
  football = 26 →
  tennis = 20 →
  both = 17 →
  total - (football + tennis - both) = 9 :=
by
  sorry

end students_playing_neither_l543_54379


namespace min_ratio_case1_min_ratio_case2_min_ratio_case3_min_ratio_case4_l543_54357

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the ratio function
def ratio (n : ℕ) : ℚ := n / (sumOfDigits n)

-- Theorem for case (i)
theorem min_ratio_case1 :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ratio n ≥ 19/10 := by sorry

-- Theorem for case (ii)
theorem min_ratio_case2 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → ratio n ≥ 119/11 := by sorry

-- Theorem for case (iii)
theorem min_ratio_case3 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → ratio n ≥ 1119/12 := by sorry

-- Theorem for case (iv)
theorem min_ratio_case4 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 → ratio n ≥ 11119/13 := by sorry

end min_ratio_case1_min_ratio_case2_min_ratio_case3_min_ratio_case4_l543_54357


namespace circle_area_equilateral_triangle_l543_54398

/-- The area of a circle circumscribing an equilateral triangle with side length 4 is 16π/3 -/
theorem circle_area_equilateral_triangle :
  let s : ℝ := 4  -- side length of the equilateral triangle
  let r : ℝ := s / Real.sqrt 3  -- radius of the circumscribed circle
  let A : ℝ := π * r^2  -- area of the circle
  A = 16 * π / 3 := by
  sorry

end circle_area_equilateral_triangle_l543_54398


namespace sqrt_198_between_14_and_15_l543_54388

theorem sqrt_198_between_14_and_15 : 14 < Real.sqrt 198 ∧ Real.sqrt 198 < 15 := by
  sorry

end sqrt_198_between_14_and_15_l543_54388


namespace correct_prime_sum_l543_54350

def isPrime (n : ℕ) : Prop :=
  ∃ m : ℕ, n + 2 = 2^m

def primeSum : ℕ := sorry

theorem correct_prime_sum : primeSum = 2026 := by sorry

end correct_prime_sum_l543_54350


namespace no_extreme_value_at_negative_one_increasing_function_p_range_l543_54327

def f (p : ℝ) (x : ℝ) : ℝ := x^3 + 3*p*x^2 + 3*p*x + 1

theorem no_extreme_value_at_negative_one (p : ℝ) :
  ¬∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x + 1| ∧ |x + 1| < ε → f p x ≤ f p (-1) ∨ f p x ≥ f p (-1) :=
sorry

theorem increasing_function_p_range :
  ∀ (p : ℝ), (∀ (x y : ℝ), -1 < x ∧ x < y → f p x < f p y) ↔ 0 ≤ p ∧ p ≤ 1 :=
sorry

end no_extreme_value_at_negative_one_increasing_function_p_range_l543_54327


namespace chairs_per_row_l543_54368

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (h1 : total_chairs = 432) (h2 : num_rows = 27) :
  total_chairs / num_rows = 16 := by
  sorry

end chairs_per_row_l543_54368


namespace cubic_difference_over_difference_l543_54353

theorem cubic_difference_over_difference (r s : ℝ) : 
  3 * r^2 - 4 * r - 12 = 0 →
  3 * s^2 - 4 * s - 12 = 0 →
  (9 * r^3 - 9 * s^3) / (r - s) = 52 := by
sorry

end cubic_difference_over_difference_l543_54353


namespace pizza_night_theorem_l543_54351

/-- Pizza night problem -/
theorem pizza_night_theorem 
  (small_pizza_slices : Nat) 
  (medium_pizza_slices : Nat) 
  (large_pizza_slices : Nat)
  (phil_eaten : Nat)
  (andre_eaten : Nat)
  (phil_ratio : Nat)
  (andre_ratio : Nat)
  (h1 : small_pizza_slices = 8)
  (h2 : medium_pizza_slices = 10)
  (h3 : large_pizza_slices = 14)
  (h4 : phil_eaten = 10)
  (h5 : andre_eaten = 12)
  (h6 : phil_ratio = 3)
  (h7 : andre_ratio = 2) :
  let total_slices := small_pizza_slices + 2 * medium_pizza_slices + large_pizza_slices
  let eaten_slices := phil_eaten + andre_eaten
  let remaining_slices := total_slices - eaten_slices
  let total_ratio := phil_ratio + andre_ratio
  let phil_share := (phil_ratio * remaining_slices) / total_ratio
  let andre_share := (andre_ratio * remaining_slices) / total_ratio
  remaining_slices = 20 ∧ phil_share = 12 ∧ andre_share = 8 := by
  sorry

#check pizza_night_theorem

end pizza_night_theorem_l543_54351


namespace initial_kids_on_soccer_field_l543_54343

theorem initial_kids_on_soccer_field (initial_kids final_kids joined_kids : ℕ) :
  final_kids = initial_kids + joined_kids →
  joined_kids = 22 →
  final_kids = 36 →
  initial_kids = 14 := by
sorry

end initial_kids_on_soccer_field_l543_54343


namespace events_mutually_exclusive_not_opposite_l543_54383

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the events
def EventAGetsRed (d : Distribution) : Prop := d Person.A = Card.Red
def EventBGetsBlue (d : Distribution) : Prop := d Person.B = Card.Blue

-- State the theorem
theorem events_mutually_exclusive_not_opposite :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventAGetsRed d ∧ EventBGetsBlue d)) ∧
  -- The events are not opposite
  (∃ d : Distribution, ¬EventAGetsRed d ∧ ¬EventBGetsBlue d) :=
by sorry

end events_mutually_exclusive_not_opposite_l543_54383


namespace negative_third_greater_than_negative_half_l543_54340

theorem negative_third_greater_than_negative_half : -1/3 > -1/2 := by
  sorry

end negative_third_greater_than_negative_half_l543_54340


namespace odd_expressions_l543_54374

theorem odd_expressions (p q : ℕ) 
  (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) : 
  Odd (5*p^2 + 2*q^2) ∧ Odd (p^2 + p*q + q^2) := by
  sorry

end odd_expressions_l543_54374


namespace age_difference_is_28_l543_54324

/-- The age difference between a man and his son -/
def ageDifference (sonAge manAge : ℕ) : ℕ := manAge - sonAge

/-- Prove that the age difference between a man and his son is 28 years -/
theorem age_difference_is_28 :
  ∃ (sonAge manAge : ℕ),
    sonAge = 26 ∧
    manAge + 2 = 2 * (sonAge + 2) ∧
    ageDifference sonAge manAge = 28 := by
  sorry

end age_difference_is_28_l543_54324


namespace tangent_line_equation_curve_passes_through_point_l543_54365

/-- The equation of the tangent line to the curve y = x^3 at the point (1,1) -/
theorem tangent_line_equation :
  ∃ (a b c : ℝ), (a * 1 + b * 1 + c = 0) ∧
  (∀ (x y : ℝ), y = x^3 → (x - 1)^2 + (y - 1)^2 ≤ (a * x + b * y + c)^2) ∧
  ((a = 3 ∧ b = -1 ∧ c = -2) ∨ (a = 3 ∧ b = -4 ∧ c = 1)) :=
by sorry

/-- The curve y = x^3 passes through the point (1,1) -/
theorem curve_passes_through_point :
  (1 : ℝ)^3 = 1 :=
by sorry

end tangent_line_equation_curve_passes_through_point_l543_54365


namespace vector_BC_proof_l543_54308

def A : ℝ × ℝ := (0, 0)  -- Assuming A as the origin for simplicity
def B : ℝ × ℝ := (2, 4)
def C : ℝ × ℝ := (1, 3)

def vector_AB : ℝ × ℝ := B
def vector_AC : ℝ × ℝ := C

theorem vector_BC_proof :
  (C.1 - B.1, C.2 - B.2) = (-1, -1) := by
  sorry

end vector_BC_proof_l543_54308


namespace multiplication_puzzle_l543_54317

theorem multiplication_puzzle :
  ∀ (G L D E N : ℕ),
    G ≠ L ∧ G ≠ D ∧ G ≠ E ∧ G ≠ N ∧
    L ≠ D ∧ L ≠ E ∧ L ≠ N ∧
    D ≠ E ∧ D ≠ N ∧
    E ≠ N ∧
    1 ≤ G ∧ G ≤ 9 ∧
    1 ≤ L ∧ L ≤ 9 ∧
    1 ≤ D ∧ D ≤ 9 ∧
    1 ≤ E ∧ E ≤ 9 ∧
    1 ≤ N ∧ N ≤ 9 ∧
    100000 * G + 40000 + 1000 * L + 100 * D + 10 * E + N = 
    (100000 * D + 10000 * E + 1000 * N + 100 * G + 40 + L) * 6 →
    G = 1 ∧ L = 2 ∧ D = 8 ∧ E = 5 ∧ N = 7 :=
by sorry

end multiplication_puzzle_l543_54317


namespace complex_multiplication_imaginary_zero_l543_54375

theorem complex_multiplication_imaginary_zero (a : ℝ) :
  (Complex.I * (a + Complex.I) + (1 : ℂ) * (a + Complex.I)).im = 0 → a = -1 := by
  sorry

end complex_multiplication_imaginary_zero_l543_54375


namespace derivative_lg_over_x_l543_54334

open Real

noncomputable def lg (x : ℝ) : ℝ := log x / log 10

theorem derivative_lg_over_x (x : ℝ) (h : x > 0) :
  deriv (λ x => lg x / x) x = (1 - log 10 * lg x) / (x^2 * log 10) :=
by sorry

end derivative_lg_over_x_l543_54334


namespace percentage_calculation_l543_54301

theorem percentage_calculation (P : ℝ) : 
  (0.15 * 0.30 * (P / 100) * 4400 = 99) → P = 50 := by
  sorry

end percentage_calculation_l543_54301


namespace two_red_one_spade_probability_l543_54371

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (red_suits : Nat)
  (black_suits : Nat)

/-- Calculates the probability of drawing two red cards followed by a spade -/
def probability_two_red_one_spade (d : Deck) : Rat :=
  if d.total_cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ d.red_suits = 2 ∧ d.black_suits = 2
  then 13 / 204
  else 0

/-- Theorem stating the probability of drawing two red cards followed by a spade from a standard deck -/
theorem two_red_one_spade_probability (d : Deck) :
  d.total_cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ d.red_suits = 2 ∧ d.black_suits = 2 →
  probability_two_red_one_spade d = 13 / 204 := by
  sorry

end two_red_one_spade_probability_l543_54371


namespace divisor_calculation_l543_54347

theorem divisor_calculation (dividend quotient remainder : ℚ) :
  dividend = 13/3 →
  quotient = -61 →
  remainder = -19 →
  ∃ divisor : ℚ, dividend = divisor * quotient + remainder ∧ divisor = -70/183 :=
by
  sorry

end divisor_calculation_l543_54347


namespace complex_multiplication_l543_54309

theorem complex_multiplication (i : ℂ) : i * i = -1 → -i * (1 - 2*i) = -2 - i := by
  sorry

end complex_multiplication_l543_54309


namespace average_increase_is_three_l543_54348

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the increase in average after a new inning -/
def averageIncrease (b : Batsman) (newRuns : ℕ) : ℚ :=
  let newAverage := (b.totalRuns + newRuns) / (b.innings + 1)
  newAverage - b.average

/-- The theorem to be proved -/
theorem average_increase_is_three :
  ∀ (b : Batsman),
    b.innings = 16 →
    (b.totalRuns + 84) / 17 = 36 →
    averageIncrease b 84 = 3 := by
  sorry

end average_increase_is_three_l543_54348


namespace octal_to_binary_127_l543_54303

theorem octal_to_binary_127 : 
  (1 * 8^2 + 2 * 8^1 + 7 * 8^0 : ℕ) = (1 * 2^6 + 0 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 : ℕ) := by
  sorry

end octal_to_binary_127_l543_54303


namespace red_balloons_total_l543_54315

/-- The number of red balloons Sara has -/
def sara_red : ℕ := 31

/-- The number of red balloons Sandy has -/
def sandy_red : ℕ := 24

/-- The total number of red balloons Sara and Sandy have -/
def total_red : ℕ := sara_red + sandy_red

theorem red_balloons_total : total_red = 55 := by
  sorry

end red_balloons_total_l543_54315


namespace no_linear_term_implies_m_equals_negative_three_l543_54358

theorem no_linear_term_implies_m_equals_negative_three (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 3) = a * x^2 + b) → m = -3 := by
  sorry

end no_linear_term_implies_m_equals_negative_three_l543_54358


namespace probability_diamond_then_ace_is_one_fiftytwo_l543_54392

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of diamond cards in a standard deck -/
def DiamondCards : ℕ := 13

/-- Represents the number of ace cards in a standard deck -/
def AceCards : ℕ := 4

/-- The probability of drawing a diamond as the first card and an ace as the second card -/
def probability_diamond_then_ace : ℚ :=
  (DiamondCards : ℚ) / StandardDeck * AceCards / (StandardDeck - 1)

theorem probability_diamond_then_ace_is_one_fiftytwo :
  probability_diamond_then_ace = 1 / StandardDeck :=
sorry

end probability_diamond_then_ace_is_one_fiftytwo_l543_54392


namespace muirhead_inequality_inequality_chain_l543_54397

/-- Symmetric mean function -/
def T (α : List ℝ) (a b c : ℝ) : ℝ := sorry

/-- Majorization relation -/
def Majorizes (α β : List ℝ) : Prop := sorry

theorem muirhead_inequality {α β : List ℝ} {a b c : ℝ} 
  (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : Majorizes α β) :
  T β a b c ≤ T α a b c := sorry

/-- Main theorem to prove -/
theorem inequality_chain (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) : 
  T [2, 1, 1] a b c ≤ T [3, 1, 0] a b c ∧ T [3, 1, 0] a b c ≤ T [4, 0, 0] a b c := by
  sorry

end muirhead_inequality_inequality_chain_l543_54397


namespace smallest_positive_number_l543_54393

theorem smallest_positive_number (a b c d e : ℝ) :
  a = 15 - 4 * Real.sqrt 14 ∧
  b = 4 * Real.sqrt 14 - 15 ∧
  c = 20 - 6 * Real.sqrt 15 ∧
  d = 60 - 12 * Real.sqrt 31 ∧
  e = 12 * Real.sqrt 31 - 60 →
  (0 < a ∧ a ≤ b ∧ a ≤ c ∧ a ≤ d ∧ a ≤ e) ∨
  (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0 ∧ d ≤ 0 ∧ e ≤ 0) :=
by sorry

end smallest_positive_number_l543_54393


namespace book_arrangement_problem_l543_54381

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) (different : ℕ) (adjacent_pair : ℕ) : ℕ :=
  (Nat.factorial (total - identical + 1 - adjacent_pair + 1) * Nat.factorial adjacent_pair) / 
  Nat.factorial identical

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem book_arrangement_problem : 
  arrange_books 7 3 4 2 = 240 := by
  sorry

end book_arrangement_problem_l543_54381


namespace always_has_real_roots_unique_integer_m_for_distinct_positive_integer_roots_l543_54322

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := m * x^2 - (m + 2) * x + 2

-- Part I: The equation always has real roots
theorem always_has_real_roots :
  ∀ m : ℝ, ∃ x : ℝ, quadratic_equation m x = 0 :=
sorry

-- Part II: Only m = 1 gives two distinct positive integer roots
theorem unique_integer_m_for_distinct_positive_integer_roots :
  ∀ m : ℤ, (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≠ y ∧
    quadratic_equation (m : ℝ) (x : ℝ) = 0 ∧
    quadratic_equation (m : ℝ) (y : ℝ) = 0) ↔ m = 1 :=
sorry

end always_has_real_roots_unique_integer_m_for_distinct_positive_integer_roots_l543_54322


namespace only_football_fans_l543_54313

/-- Represents the number of people in different categories in a class --/
structure ClassPreferences where
  total : Nat
  baseballAndFootball : Nat
  onlyBaseball : Nat
  neitherSport : Nat
  onlyFootball : Nat

/-- The theorem stating the number of people who only like football --/
theorem only_football_fans (c : ClassPreferences) : c.onlyFootball = 3 :=
  by
  have h1 : c.total = 16 := by sorry
  have h2 : c.baseballAndFootball = 5 := by sorry
  have h3 : c.onlyBaseball = 2 := by sorry
  have h4 : c.neitherSport = 6 := by sorry
  have h5 : c.total = c.baseballAndFootball + c.onlyBaseball + c.onlyFootball + c.neitherSport := by sorry
  sorry

#check only_football_fans

end only_football_fans_l543_54313


namespace solve_for_y_l543_54366

theorem solve_for_y (x z : ℝ) (h1 : x^2 * z - x * z^2 = 6) (h2 : x = -2) (h3 : z = 1) : 
  ∃ y : ℝ, x^2 * y * z - x * y * z^2 = 6 ∧ y = 1 := by
sorry

end solve_for_y_l543_54366


namespace penny_count_l543_54302

/-- Proves that given 4 nickels, 3 dimes, and a total value of $0.59, the number of pennies is 9 -/
theorem penny_count (nickels : ℕ) (dimes : ℕ) (total_cents : ℕ) (pennies : ℕ) : 
  nickels = 4 → 
  dimes = 3 → 
  total_cents = 59 → 
  5 * nickels + 10 * dimes + pennies = total_cents → 
  pennies = 9 := by
sorry

end penny_count_l543_54302


namespace calzone_time_proof_l543_54399

def calzone_time_calculation (onion_time garlic_time knead_time rest_time assemble_time : ℕ) : Prop :=
  (garlic_time = onion_time / 4) ∧
  (rest_time = 2 * knead_time) ∧
  (assemble_time = (knead_time + rest_time) / 10) ∧
  (onion_time + garlic_time + knead_time + rest_time + assemble_time = 124)

theorem calzone_time_proof :
  ∃ (onion_time garlic_time knead_time rest_time assemble_time : ℕ),
    onion_time = 20 ∧
    knead_time = 30 ∧
    calzone_time_calculation onion_time garlic_time knead_time rest_time assemble_time :=
by
  sorry

end calzone_time_proof_l543_54399


namespace set_inclusion_equivalence_l543_54321

theorem set_inclusion_equivalence (a : ℝ) : 
  let A := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 22}
  (∃ x, x ∈ A) → (A ⊆ A ∩ B ↔ 6 ≤ a ∧ a ≤ 9) := by
sorry

end set_inclusion_equivalence_l543_54321


namespace highest_power_of_three_in_M_l543_54325

def M : ℕ := sorry

theorem highest_power_of_three_in_M : 
  (∃ k : ℕ, M = 3 * k) ∧ ¬(∃ k : ℕ, M = 9 * k) := by sorry

end highest_power_of_three_in_M_l543_54325


namespace average_food_expense_percentage_l543_54312

/-- Calculate the average percentage of income spent on food over two months --/
theorem average_food_expense_percentage (jan_income feb_income : ℚ)
  (jan_petrol feb_petrol : ℚ) : 
  jan_income = 3000 →
  feb_income = 4000 →
  jan_petrol = 300 →
  feb_petrol = 400 →
  let jan_remaining := jan_income - jan_petrol
  let feb_remaining := feb_income - feb_petrol
  let jan_rent := jan_remaining * (14 / 100)
  let feb_rent := feb_remaining * (14 / 100)
  let jan_clothing := jan_income * (10 / 100)
  let feb_clothing := feb_income * (10 / 100)
  let jan_utility := jan_income * (5 / 100)
  let feb_utility := feb_income * (5 / 100)
  let jan_food := jan_remaining - jan_rent - jan_clothing - jan_utility
  let feb_food := feb_remaining - feb_rent - feb_clothing - feb_utility
  let total_food := jan_food + feb_food
  let total_income := jan_income + feb_income
  let avg_food_percentage := (total_food / total_income) * 100
  avg_food_percentage = 62.4 := by
  sorry

end average_food_expense_percentage_l543_54312


namespace random_events_l543_54356

-- Define the type for events
inductive Event
  | CoinToss
  | ChargeAttraction
  | WaterFreezing
  | DiceRoll

-- Define a function to check if an event is random
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.CoinToss => true
  | Event.ChargeAttraction => false
  | Event.WaterFreezing => false
  | Event.DiceRoll => true

-- Theorem stating which events are random
theorem random_events :
  (isRandomEvent Event.CoinToss) ∧
  (¬isRandomEvent Event.ChargeAttraction) ∧
  (¬isRandomEvent Event.WaterFreezing) ∧
  (isRandomEvent Event.DiceRoll) := by
  sorry

#check random_events

end random_events_l543_54356


namespace sandro_children_l543_54390

/-- The number of sons Sandro has -/
def num_sons : ℕ := 3

/-- The ratio of daughters to sons -/
def daughter_son_ratio : ℕ := 6

/-- The number of daughters Sandro has -/
def num_daughters : ℕ := daughter_son_ratio * num_sons

/-- The total number of children Sandro has -/
def total_children : ℕ := num_daughters + num_sons

theorem sandro_children : total_children = 21 := by
  sorry

end sandro_children_l543_54390


namespace solution_set_empty_l543_54391

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

-- Define the constants a and b
def a : ℝ := 2
def b : ℝ := -3

-- State the theorem
theorem solution_set_empty :
  ∀ x : ℝ, f a (a * x + b) ≠ 0 :=
sorry

end solution_set_empty_l543_54391


namespace inequality_proof_l543_54354

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b^2 / a + c^2 / b + a^2 / c ≥ a + b + c := by
  sorry

end inequality_proof_l543_54354


namespace sum_of_ratios_ge_two_l543_54362

theorem sum_of_ratios_ge_two (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / (c + d) + b / (d + a) + c / (a + b) + d / (b + c) ≥ 2 := by
  sorry

end sum_of_ratios_ge_two_l543_54362


namespace marathon_remainder_l543_54346

/-- Represents the distance of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a runner's total distance in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ

/-- Converts a given number of yards to miles and remaining yards -/
def yardsToMilesAndYards (totalYards : ℕ) : TotalDistance :=
  { miles := totalYards / 1760,
    yards := totalYards % 1760 }

theorem marathon_remainder (marathonDistance : Marathon) (numMarathons : ℕ) : 
  marathonDistance.miles = 26 →
  marathonDistance.yards = 395 →
  numMarathons = 15 →
  (yardsToMilesAndYards (numMarathons * (marathonDistance.miles * 1760 + marathonDistance.yards))).yards = 645 := by
  sorry

#check marathon_remainder

end marathon_remainder_l543_54346


namespace cone_roll_ratio_sum_l543_54373

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height
  r_pos : r > 0
  h_pos : h > 0

/-- Checks if a number is not a multiple of any prime squared -/
def not_multiple_of_prime_squared (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ n)

/-- Main theorem -/
theorem cone_roll_ratio_sum (cone : RightCircularCone) 
    (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0)
    (h_ratio : cone.h / cone.r = m * Real.sqrt n)
    (h_rotations : (2 * Real.pi * Real.sqrt (cone.r^2 + cone.h^2)) = 40 * Real.pi * cone.r)
    (h_not_multiple : not_multiple_of_prime_squared n) :
    m + n = 136 := by
  sorry

end cone_roll_ratio_sum_l543_54373


namespace minimum_rental_fee_for_360_people_l543_54341

/-- Represents a bus type with its seat capacity and rental fee -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the minimum rental fee for transporting a given number of people -/
def minimumRentalFee (totalPeople : ℕ) (typeA typeB : BusType) : ℕ :=
  sorry

theorem minimum_rental_fee_for_360_people :
  let typeA : BusType := ⟨40, 400⟩
  let typeB : BusType := ⟨50, 480⟩
  minimumRentalFee 360 typeA typeB = 3520 := by
  sorry

end minimum_rental_fee_for_360_people_l543_54341


namespace kelsey_travel_time_l543_54372

theorem kelsey_travel_time (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 400)
  (h2 : speed1 = 25)
  (h3 : speed2 = 40) : 
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 13 := by
  sorry

end kelsey_travel_time_l543_54372


namespace geometric_sequence_common_ratio_l543_54330

/-- Given a geometric sequence {a_n} where a_3 = 6 and the sum of the first three terms S_3 = 18,
    prove that the common ratio q is either 1 or -1/2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 3 = 6 →                     -- Third term is 6
  a 1 + a 2 + a 3 = 18 →        -- Sum of first three terms is 18
  q = 1 ∨ q = -1/2 :=
by sorry

end geometric_sequence_common_ratio_l543_54330


namespace cone_lateral_surface_area_l543_54332

theorem cone_lateral_surface_area 
  (base_radius : ℝ) 
  (height : ℝ) 
  (lateral_surface_area : ℝ) 
  (h1 : base_radius = 3) 
  (h2 : height = 4) : 
  lateral_surface_area = 15 * Real.pi := by
  sorry

end cone_lateral_surface_area_l543_54332


namespace index_card_area_l543_54386

theorem index_card_area (length width : ℝ) : 
  length = 8 ∧ width = 3 →
  (∃ new_length, new_length = length - 2 ∧ new_length * width = 18) →
  (width - 2) * length = 8 :=
by sorry

end index_card_area_l543_54386


namespace good_numbers_in_set_l543_54311

-- Define what a "good number" is
def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), Function.Bijective a ∧
    ∀ k : Fin n, ∃ m : ℕ, (k.val + 1 + (a k).val + 1 : ℕ) = m * m

-- Theorem statement
theorem good_numbers_in_set :
  isGoodNumber 11 = false ∧
  isGoodNumber 13 = true ∧
  isGoodNumber 15 = true ∧
  isGoodNumber 17 = true ∧
  isGoodNumber 19 = true :=
by sorry

end good_numbers_in_set_l543_54311


namespace largest_angle_of_convex_hexagon_consecutive_angles_l543_54385

-- Define a type for convex hexagons with consecutive integer angle measures
structure ConvexHexagonConsecutiveAngles where
  -- The smallest angle measure
  smallest_angle : ℕ
  -- Ensure the hexagon is convex (all angles are less than 180°)
  convex : smallest_angle + 5 < 180

-- Define the sum of interior angles of a hexagon
def hexagon_angle_sum : ℕ := 720

-- Theorem statement
theorem largest_angle_of_convex_hexagon_consecutive_angles 
  (h : ConvexHexagonConsecutiveAngles) : 
  h.smallest_angle + 5 = 122 :=
sorry

end largest_angle_of_convex_hexagon_consecutive_angles_l543_54385


namespace sum_of_series_equals_25_16_l543_54306

theorem sum_of_series_equals_25_16 : 
  (∑' n, n / 5^n) + (∑' n, (1 / 5)^n) = 25 / 16 :=
by sorry

end sum_of_series_equals_25_16_l543_54306


namespace total_people_on_boats_l543_54323

/-- The number of boats in the lake -/
def num_boats : ℕ := 5

/-- The number of people on each boat -/
def people_per_boat : ℕ := 3

/-- The total number of people on boats in the lake -/
def total_people : ℕ := num_boats * people_per_boat

theorem total_people_on_boats : total_people = 15 := by
  sorry

end total_people_on_boats_l543_54323


namespace lawn_length_l543_54316

/-- Given a rectangular lawn with area 20 square feet and width 5 feet, prove its length is 4 feet. -/
theorem lawn_length (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 20 → width = 5 → area = length * width → length = 4 := by
  sorry

end lawn_length_l543_54316


namespace product_of_ratios_l543_54363

/-- Given three pairs of real numbers satisfying specific equations, 
    prove that the product of their ratios equals a specific value. -/
theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2007) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2006)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2007) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2006)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2007) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2006) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -2011/2006 := by
  sorry

end product_of_ratios_l543_54363


namespace umar_age_is_10_l543_54380

-- Define the ages as natural numbers
def ali_age : ℕ := 8
def age_difference : ℕ := 3
def umar_age_multiplier : ℕ := 2

-- Theorem to prove
theorem umar_age_is_10 :
  let yusaf_age := ali_age - age_difference
  let umar_age := umar_age_multiplier * yusaf_age
  umar_age = 10 := by sorry

end umar_age_is_10_l543_54380


namespace sequence_property_l543_54384

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n > 0) →
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → a n * (n * a n - a (n + 1)) = (n + 1) * (a (n + 1))^2) →
  ∀ n : ℕ, n > 0 → a n = 1 / n :=
by sorry

end sequence_property_l543_54384


namespace ahn_max_number_l543_54344

theorem ahn_max_number : ∃ (max : ℕ), max = 700 ∧ 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 2 * (500 - n - 50) ≤ max :=
by sorry

end ahn_max_number_l543_54344


namespace min_value_quadratic_form_l543_54377

theorem min_value_quadratic_form (a b : ℝ) (h : 4 ≤ a^2 + b^2 ∧ a^2 + b^2 ≤ 9) :
  2 ≤ a^2 - a*b + b^2 ∧ ∃ (x y : ℝ), 4 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 9 ∧ x^2 - x*y + y^2 = 2 := by
  sorry

end min_value_quadratic_form_l543_54377


namespace jake_has_nine_peaches_l543_54335

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 16

/-- The number of peaches Jake has fewer than Steven -/
def jake_fewer_than_steven : ℕ := 7

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - jake_fewer_than_steven

/-- Theorem: Jake has 9 peaches -/
theorem jake_has_nine_peaches : jake_peaches = 9 := by
  sorry

end jake_has_nine_peaches_l543_54335


namespace cubic_root_sum_l543_54369

theorem cubic_root_sum (p q r : ℝ) : 
  0 < p ∧ p < 1 ∧ 
  0 < q ∧ q < 1 ∧ 
  0 < r ∧ r < 1 ∧ 
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  30 * p^3 - 50 * p^2 + 22 * p - 1 = 0 ∧
  30 * q^3 - 50 * q^2 + 22 * q - 1 = 0 ∧
  30 * r^3 - 50 * r^2 + 22 * r - 1 = 0 →
  1 / (1 - p) + 1 / (1 - q) + 1 / (1 - r) = 12 := by
sorry

end cubic_root_sum_l543_54369
