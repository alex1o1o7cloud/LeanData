import Mathlib

namespace lock_code_difference_l2237_223761

def is_valid_code (a b c : Nat) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (b * b) = (a * c * c)

def code_value (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

theorem lock_code_difference : 
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : Nat),
    is_valid_code a₁ b₁ c₁ ∧
    is_valid_code a₂ b₂ c₂ ∧
    (∀ a b c, is_valid_code a b c → 
      code_value a b c ≤ code_value a₁ b₁ c₁ ∧
      code_value a b c ≥ code_value a₂ b₂ c₂) ∧
    code_value a₁ b₁ c₁ - code_value a₂ b₂ c₂ = 541 :=
by sorry

end lock_code_difference_l2237_223761


namespace cos_sixty_degrees_l2237_223742

theorem cos_sixty_degrees : Real.cos (60 * π / 180) = 1 / 2 := by
  sorry

end cos_sixty_degrees_l2237_223742


namespace sand_collection_total_weight_l2237_223786

theorem sand_collection_total_weight (eden_buckets mary_buckets iris_buckets : ℕ) 
  (sand_weight_per_bucket : ℕ) :
  eden_buckets = 4 →
  mary_buckets = eden_buckets + 3 →
  iris_buckets = mary_buckets - 1 →
  sand_weight_per_bucket = 2 →
  (eden_buckets + mary_buckets + iris_buckets) * sand_weight_per_bucket = 34 :=
by
  sorry

end sand_collection_total_weight_l2237_223786


namespace cylindrical_to_cartesian_l2237_223718

/-- Given a point P in cylindrical coordinates (r, θ, z) = (√2, π/4, 1),
    prove that its Cartesian coordinates (x, y, z) are (1, 1, 1). -/
theorem cylindrical_to_cartesian :
  let r : ℝ := Real.sqrt 2
  let θ : ℝ := π / 4
  let z : ℝ := 1
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y, z) = (1, 1, 1) := by sorry

end cylindrical_to_cartesian_l2237_223718


namespace some_frames_are_not_tars_l2237_223717

universe u

-- Define the types
variable (Tar Kite Rope Frame : Type u)

-- Define the relations
variable (is_tar : Tar → Prop)
variable (is_kite : Kite → Prop)
variable (is_rope : Rope → Prop)
variable (is_frame : Frame → Prop)

-- Hypotheses
variable (h1 : ∀ t : Tar, ∃ k : Kite, is_kite k)
variable (h2 : ∀ k : Kite, ∀ r : Rope, ¬(is_kite k ∧ is_rope r))
variable (h3 : ∃ r : Rope, ∃ f : Frame, is_rope r ∧ is_frame f)

-- Theorem to prove
theorem some_frames_are_not_tars :
  ∃ f : Frame, ¬∃ t : Tar, is_frame f ∧ is_tar t :=
sorry

end some_frames_are_not_tars_l2237_223717


namespace power_multiplication_l2237_223782

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l2237_223782


namespace equation_solution_l2237_223755

theorem equation_solution : 
  ∃ x : ℝ, (x / (x - 1) = (x - 3) / (2*x - 2)) ∧ (x = -3) := by sorry

end equation_solution_l2237_223755


namespace min_h_10_l2237_223712

/-- A function is expansive if f(x) + f(y) > x^2 + y^2 for all positive integers x and y -/
def Expansive (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y > (x.val : ℤ)^2 + (y.val : ℤ)^2

/-- The sum of h(1) to h(15) -/
def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (λ i => h ⟨i + 1, by linarith⟩)

/-- The theorem statement -/
theorem min_h_10 (h : ℕ+ → ℤ) (hExpansive : Expansive h) (hMinSum : ∀ g : ℕ+ → ℤ, Expansive g → SumH g ≥ SumH h) :
  h ⟨10, by norm_num⟩ ≥ 125 := by
  sorry

end min_h_10_l2237_223712


namespace monk_problem_l2237_223775

theorem monk_problem (total_mantou total_monks : ℕ) 
  (big_monk_consumption small_monk_consumption : ℚ) :
  total_mantou = 100 →
  total_monks = 100 →
  big_monk_consumption = 1 →
  small_monk_consumption = 1/3 →
  ∃ (big_monks small_monks : ℕ),
    big_monks + small_monks = total_monks ∧
    big_monks * big_monk_consumption + small_monks * small_monk_consumption = total_mantou ∧
    big_monks = 25 ∧
    small_monks = 75 := by
  sorry

end monk_problem_l2237_223775


namespace fraction_equality_l2237_223791

theorem fraction_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  (a + b) / a = 7 / 4 := by
  sorry

end fraction_equality_l2237_223791


namespace divisibility_of_difference_l2237_223770

theorem divisibility_of_difference : 43^43 - 17^17 ≡ 0 [ZMOD 10] := by
  sorry

end divisibility_of_difference_l2237_223770


namespace external_tangent_intercept_l2237_223789

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in slope-intercept form --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Checks if a line is external tangent to two circles --/
def isExternalTangent (l : Line) (c1 c2 : Circle) : Prop :=
  sorry

theorem external_tangent_intercept :
  let c1 : Circle := { center := (3, -2), radius := 3 }
  let c2 : Circle := { center := (15, 8), radius := 8 }
  ∀ l : Line,
    l.slope > 0 →
    isExternalTangent l c1 c2 →
    l.intercept = 720 / 11 :=
sorry

end external_tangent_intercept_l2237_223789


namespace inequality_holds_in_intervals_l2237_223767

theorem inequality_holds_in_intervals (a b : ℝ) : 
  (((0 ≤ a ∧ a < b ∧ b ≤ π/2) ∨ (π ≤ a ∧ a < b ∧ b ≤ 3*π/2)) → 
   (a - Real.sin a < b - Real.sin b)) := by
  sorry

end inequality_holds_in_intervals_l2237_223767


namespace trapezoid_solution_l2237_223779

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  a : ℝ  -- Length of the shorter parallel side
  h : ℝ  -- Height of the trapezoid
  area : ℝ -- Area of the trapezoid

/-- Properties of the trapezoid -/
def trapezoid_properties (t : Trapezoid) : Prop :=
  t.h = (2 * t.a + 3) / 2 ∧
  t.area = t.a^2 + 3 * t.a + 9 / 4 ∧
  t.area = 2 * t.a^2 - 7.75

/-- Theorem stating the solution to the trapezoid problem -/
theorem trapezoid_solution (t : Trapezoid) (h : trapezoid_properties t) :
  t.a = 5 ∧ t.a + 3 = 8 ∧ t.h = 6.5 := by
  sorry


end trapezoid_solution_l2237_223779


namespace smallest_translation_l2237_223741

open Real

theorem smallest_translation (φ : ℝ) : φ > 0 ∧ 
  (∀ x : ℝ, sin (2 * (x + φ)) = cos (2 * x - π / 3)) →
  φ = π / 12 :=
by
  sorry

end smallest_translation_l2237_223741


namespace mike_red_notebooks_l2237_223715

/-- Represents the number of red notebooks Mike bought -/
def red_notebooks : ℕ := sorry

/-- Represents the number of blue notebooks Mike bought -/
def blue_notebooks : ℕ := sorry

/-- The total cost of all notebooks -/
def total_cost : ℕ := 37

/-- The total number of notebooks -/
def total_notebooks : ℕ := 12

/-- The cost of each red notebook -/
def red_cost : ℕ := 4

/-- The number of green notebooks -/
def green_notebooks : ℕ := 2

/-- The cost of each green notebook -/
def green_cost : ℕ := 2

/-- The cost of each blue notebook -/
def blue_cost : ℕ := 3

theorem mike_red_notebooks : 
  red_notebooks = 3 ∧
  red_notebooks + green_notebooks + blue_notebooks = total_notebooks ∧
  red_notebooks * red_cost + green_notebooks * green_cost + blue_notebooks * blue_cost = total_cost :=
sorry

end mike_red_notebooks_l2237_223715


namespace additional_grazing_area_l2237_223777

/-- The additional grassy ground area a calf can graze after increasing rope length -/
theorem additional_grazing_area (initial_length new_length obstacle_length obstacle_width : ℝ) 
  (h1 : initial_length = 12)
  (h2 : new_length = 18)
  (h3 : obstacle_length = 4)
  (h4 : obstacle_width = 3) :
  (π * new_length^2 - obstacle_length * obstacle_width) - π * initial_length^2 = 180 * π - 12 :=
by sorry

end additional_grazing_area_l2237_223777


namespace bear_ate_54_pies_l2237_223759

/-- Represents the eating scenario of Masha and the Bear -/
structure EatingScenario where
  totalPies : ℕ
  bearRaspberrySpeed : ℕ
  bearPieSpeed : ℕ
  bearRaspberryRatio : ℕ

/-- Calculates the number of pies eaten by the Bear -/
def bearPies (scenario : EatingScenario) : ℕ :=
  sorry

/-- Theorem stating that the Bear ate 54 pies -/
theorem bear_ate_54_pies (scenario : EatingScenario) 
  (h1 : scenario.totalPies = 60)
  (h2 : scenario.bearRaspberrySpeed = 6)
  (h3 : scenario.bearPieSpeed = 3)
  (h4 : scenario.bearRaspberryRatio = 2) :
  bearPies scenario = 54 := by
  sorry

end bear_ate_54_pies_l2237_223759


namespace perimeter_is_ten_x_l2237_223768

/-- The perimeter of a figure composed of rectangular segments -/
def perimeter_of_figure (x : ℝ) (hx : x ≠ 0) : ℝ :=
  let vertical_length1 := 3 * x
  let vertical_length2 := x
  let horizontal_length1 := 2 * x
  let horizontal_length2 := x
  vertical_length1 + vertical_length2 + horizontal_length1 + horizontal_length2 + 
  (3 * x - x) + (2 * x - x)

theorem perimeter_is_ten_x (x : ℝ) (hx : x ≠ 0) :
  perimeter_of_figure x hx = 10 * x := by
  sorry

end perimeter_is_ten_x_l2237_223768


namespace rider_distances_l2237_223709

/-- The possible distances between two riders after one hour, given their initial distance and speeds -/
theorem rider_distances (initial_distance : ℝ) (speed_athos : ℝ) (speed_aramis : ℝ) :
  initial_distance = 20 ∧ speed_athos = 4 ∧ speed_aramis = 5 →
  ∃ (d₁ d₂ d₃ d₄ : ℝ),
    d₁ = 11 ∧ d₂ = 29 ∧ d₃ = 19 ∧ d₄ = 21 ∧
    ({d₁, d₂, d₃, d₄} : Set ℝ) = {
      initial_distance - (speed_athos + speed_aramis),
      initial_distance + (speed_athos + speed_aramis),
      initial_distance - (speed_aramis - speed_athos),
      initial_distance + (speed_aramis - speed_athos)
    } := by sorry

end rider_distances_l2237_223709


namespace candy_distribution_l2237_223793

/-- 
Given a group of students where each student receives a fixed number of candy pieces,
this theorem proves that the total number of candy pieces given away is equal to
the product of the number of students and the number of pieces per student.
-/
theorem candy_distribution (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : num_students = 9) 
  (h2 : pieces_per_student = 2) : 
  num_students * pieces_per_student = 18 := by
  sorry

end candy_distribution_l2237_223793


namespace inequality_proof_l2237_223765

theorem inequality_proof (x : ℝ) (h : x ≠ 1) :
  Real.sqrt (x^2 - 2*x + 2) ≥ -Real.sqrt 5 * x ↔ (-1 ≤ x ∧ x < 1) ∨ x > 1 :=
by sorry

end inequality_proof_l2237_223765


namespace pascal_row15_element4_l2237_223785

/-- Pascal's triangle element -/
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

/-- The fourth element in Row 15 of Pascal's triangle -/
def row15_element4 : ℕ := pascal 15 3

/-- Theorem: The fourth element in Row 15 of Pascal's triangle is 455 -/
theorem pascal_row15_element4 : row15_element4 = 455 := by
  sorry

end pascal_row15_element4_l2237_223785


namespace students_in_cars_l2237_223784

theorem students_in_cars (total_students : ℕ) (num_buses : ℕ) (students_per_bus : ℕ) :
  total_students = 396 →
  num_buses = 7 →
  students_per_bus = 56 →
  total_students - (num_buses * students_per_bus) = 4 :=
by sorry

end students_in_cars_l2237_223784


namespace max_y_coordinate_sin_3theta_l2237_223745

/-- The maximum y-coordinate of a point on the curve r = sin 3θ is 9/16 -/
theorem max_y_coordinate_sin_3theta : 
  let r : ℝ → ℝ := λ θ => Real.sin (3 * θ)
  let y : ℝ → ℝ := λ θ => r θ * Real.sin θ
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), y θ ≤ y θ_max ∧ y θ_max = 9/16 :=
by
  sorry


end max_y_coordinate_sin_3theta_l2237_223745


namespace product_of_difference_and_sum_of_squares_l2237_223748

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 31) : 
  a * b = 3 := by
  sorry

end product_of_difference_and_sum_of_squares_l2237_223748


namespace students_per_school_l2237_223752

theorem students_per_school (total_schools : ℕ) (total_students : ℕ) 
  (h1 : total_schools = 25) (h2 : total_students = 6175) : 
  total_students / total_schools = 247 := by
  sorry

end students_per_school_l2237_223752


namespace different_meal_combinations_l2237_223731

theorem different_meal_combinations (n : ℕ) (h : n = 12) : n * (n - 1) = 132 := by
  sorry

end different_meal_combinations_l2237_223731


namespace toms_age_ratio_l2237_223711

/-- Tom's age problem -/
theorem toms_age_ratio (T N : ℝ) : T > 0 → N > 0 → 
  (T = T - 4*N + T - 4*N + T - 4*N + T - 4*N) → -- Sum of children's ages
  (T - N = 3 * (T - 4*N)) →                     -- Relation N years ago
  T / N = 11 / 2 := by
sorry

end toms_age_ratio_l2237_223711


namespace elevator_weight_problem_l2237_223725

/-- Proves that the initial average weight of 6 people in an elevator was 156 lbs,
    given that a 7th person weighing 121 lbs entered and increased the average to 151 lbs. -/
theorem elevator_weight_problem (initial_count : Nat) (new_person_weight : Nat) (new_average : Nat) :
  initial_count = 6 →
  new_person_weight = 121 →
  new_average = 151 →
  ∃ (initial_average : Nat),
    initial_average = 156 ∧
    (initial_count * initial_average + new_person_weight) / (initial_count + 1) = new_average :=
by sorry

end elevator_weight_problem_l2237_223725


namespace abs_neg_two_eq_two_l2237_223705

theorem abs_neg_two_eq_two : |(-2 : ℝ)| = 2 := by
  sorry

end abs_neg_two_eq_two_l2237_223705


namespace transform_sin_function_l2237_223781

open Real

theorem transform_sin_function (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π/2) :
  let f : ℝ → ℝ := λ x ↦ 2 * sin (3*x + φ)
  let g : ℝ → ℝ := λ x ↦ 2 * sin (3*x) + 1
  (∀ x, f x = f (2*φ - x)) →  -- (φ, 0) is center of symmetry
  (∃ h : ℝ → ℝ, ∀ x, g x = h (f (x - π/12)) + 1) :=
by sorry

end transform_sin_function_l2237_223781


namespace sum_minus_seven_tenths_l2237_223764

theorem sum_minus_seven_tenths (a b c : ℝ) (ha : a = 34.5) (hb : b = 15.2) (hc : c = 0.7) :
  a + b - c = 49 := by
  sorry

end sum_minus_seven_tenths_l2237_223764


namespace jungkook_persimmons_jungkook_picked_8_persimmons_l2237_223706

theorem jungkook_persimmons : ℕ → Prop :=
  fun j : ℕ =>
    let h := 35  -- Hoseok's persimmons
    h = 4 * j + 3 → j = 8

-- Proof
theorem jungkook_picked_8_persimmons : jungkook_persimmons 8 := by
  sorry

end jungkook_persimmons_jungkook_picked_8_persimmons_l2237_223706


namespace cheesecakes_sold_l2237_223799

theorem cheesecakes_sold (display : ℕ) (fridge : ℕ) (left : ℕ) : 
  display + fridge - left = display - (display + fridge - left - fridge) :=
by sorry

#check cheesecakes_sold 10 15 18

end cheesecakes_sold_l2237_223799


namespace football_season_duration_l2237_223783

theorem football_season_duration (total_games : ℕ) (games_per_month : ℕ) 
  (h1 : total_games = 323) 
  (h2 : games_per_month = 19) : 
  total_games / games_per_month = 17 := by
  sorry

end football_season_duration_l2237_223783


namespace right_triangle_perimeter_l2237_223732

theorem right_triangle_perimeter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_area : (1/2) * a * b = 150) (h_leg : a = 30) : 
  a + b + c = 40 + 10 * Real.sqrt 10 := by
  sorry

end right_triangle_perimeter_l2237_223732


namespace friend_distribution_l2237_223749

theorem friend_distribution (F : ℕ) (h1 : F > 0) : 
  (100 / F : ℚ) - (100 / (F + 5) : ℚ) = 1 → F = 20 := by
  sorry

end friend_distribution_l2237_223749


namespace additional_hour_rate_is_ten_l2237_223708

/-- Represents the rental cost structure for a power tool -/
structure RentalCost where
  firstHourRate : ℝ
  additionalHourRate : ℝ
  totalHours : ℕ
  totalCost : ℝ

/-- Theorem stating that given the rental conditions, the additional hour rate is $10 -/
theorem additional_hour_rate_is_ten
  (rental : RentalCost)
  (h1 : rental.firstHourRate = 25)
  (h2 : rental.totalHours = 11)
  (h3 : rental.totalCost = 125)
  : rental.additionalHourRate = 10 := by
  sorry

#check additional_hour_rate_is_ten

end additional_hour_rate_is_ten_l2237_223708


namespace jason_shampoo_time_l2237_223795

theorem jason_shampoo_time :
  ∀ (J : ℝ),
  J > 0 →
  (1 / J + 1 / 6 = 1 / 2) →
  J = 3 :=
by sorry

end jason_shampoo_time_l2237_223795


namespace area_of_graph_l2237_223700

/-- The area enclosed by the graph of |x| + |3y| = 12 -/
def rhombus_area : ℝ := 384

/-- The equation defining the graph -/
def graph_equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

theorem area_of_graph :
  ∃ (x_intercept y_intercept : ℝ),
    x_intercept > 0 ∧
    y_intercept > 0 ∧
    graph_equation x_intercept 0 ∧
    graph_equation 0 y_intercept ∧
    rhombus_area = 4 * (x_intercept * y_intercept) :=
sorry

end area_of_graph_l2237_223700


namespace max_period_linear_recurrence_l2237_223730

/-- The maximum period of a second-order linear recurrence sequence modulo a prime -/
theorem max_period_linear_recurrence (p : Nat) (hp : Prime p) 
  (a b c d : Int) : ∃ (x : Nat → Int), 
  (x 0 = c) ∧ 
  (x 1 = d) ∧ 
  (∀ n, x (n + 2) = a * x (n + 1) + b * x n) ∧ 
  (∃ t, t ≤ p^2 - 1 ∧ 
    ∀ n ≥ p^2, (x (n + t) : ZMod p) = (x n : ZMod p)) ∧
  (∀ t' < p^2 - 1, ∃ n ≥ p^2, (x (n + t') : ZMod p) ≠ (x n : ZMod p)) :=
sorry

end max_period_linear_recurrence_l2237_223730


namespace intersection_equals_open_interval_l2237_223751

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | Real.log (x - 1) ≤ 0}

-- State the theorem
theorem intersection_equals_open_interval :
  A ∩ B = Set.Ioo 1 2 := by sorry

end intersection_equals_open_interval_l2237_223751


namespace arithmetic_sequence_sum_l2237_223756

/-- Given an arithmetic sequence {aₙ}, prove that S₁₃ = 13 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence condition
  (∀ n : ℕ, S n = (n / 2) * (a 1 + a n)) →               -- sum formula
  a 3 + a 5 + 2 * a 10 = 4 →                             -- given condition
  S 13 = 13 := by
sorry

end arithmetic_sequence_sum_l2237_223756


namespace decrease_by_percentage_decrease_80_by_150_percent_l2237_223796

theorem decrease_by_percentage (n : ℝ) (p : ℝ) : 
  n - (p / 100) * n = n * (1 - p / 100) := by sorry

theorem decrease_80_by_150_percent : 
  80 - (150 / 100) * 80 = -40 := by sorry

end decrease_by_percentage_decrease_80_by_150_percent_l2237_223796


namespace geometric_sequence_property_l2237_223719

/-- Given a geometric sequence {a_n} where a_4 = 4, prove that a_2 * a_6 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence property
  a 4 = 4 →                                            -- given condition
  a 2 * a 6 = 16 :=                                    -- conclusion to prove
by
  sorry

end geometric_sequence_property_l2237_223719


namespace min_value_and_inequality_l2237_223774

theorem min_value_and_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 2*b + 3*c = 8) →
  (∃ (m : ℝ), m = 1/a + 2/b + 3/c ∧ m ≥ 4.5 ∧ ∀ (x : ℝ), x = 1/a + 2/b + 3/c → x ≥ m) ∧
  (∃ (x : ℝ), (x = a + 1/b ∨ x = b + 1/c ∨ x = c + 1/a) ∧ x ≥ 2) :=
by sorry


end min_value_and_inequality_l2237_223774


namespace min_stamps_for_40_cents_l2237_223702

theorem min_stamps_for_40_cents :
  let stamp_values : List Nat := [5, 7]
  let target_value : Nat := 40
  ∃ (c f : Nat),
    c * stamp_values[0]! + f * stamp_values[1]! = target_value ∧
    ∀ (c' f' : Nat),
      c' * stamp_values[0]! + f' * stamp_values[1]! = target_value →
      c + f ≤ c' + f' ∧
    c + f = 6 :=
by sorry

end min_stamps_for_40_cents_l2237_223702


namespace sean_patch_profit_l2237_223794

/-- Calculates the net profit for Sean's patch business -/
theorem sean_patch_profit :
  let order_quantity : ℕ := 100
  let cost_per_patch : ℚ := 125/100
  let sell_price_per_patch : ℚ := 12
  let total_cost : ℚ := order_quantity * cost_per_patch
  let total_revenue : ℚ := order_quantity * sell_price_per_patch
  let net_profit : ℚ := total_revenue - total_cost
  net_profit = 1075 := by sorry

end sean_patch_profit_l2237_223794


namespace tangent_line_intersection_l2237_223754

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 = 4 * p.2}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -2}

-- Define a function to get tangent points on C from a point on l
def tangentPoints (E : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ C ∧ ∃ m : ℝ, (p.2 - E.2) = m * (p.1 - E.1) ∧ p.1 = 2 * m}

-- Theorem statement
theorem tangent_line_intersection (E : ℝ × ℝ) (hE : E ∈ l) :
  ∃ A B : ℝ × ℝ, A ∈ tangentPoints E ∧ B ∈ tangentPoints E ∧ A ≠ B ∧
  ∃ t : ℝ, (1 - t) • A + t • B = (0, 2) :=
sorry

end tangent_line_intersection_l2237_223754


namespace inequality_solution_set_l2237_223788

theorem inequality_solution_set (x : ℝ) : (2 * x - 1) / (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 := by
  sorry

end inequality_solution_set_l2237_223788


namespace integral_evaluation_l2237_223736

theorem integral_evaluation :
  ∫ x in (1 : ℝ)..2, (x + 1/x + 1/x^2) = 2 + Real.log 2 := by sorry

end integral_evaluation_l2237_223736


namespace inverse_function_property_l2237_223798

theorem inverse_function_property (f : ℝ → ℝ) (h_inv : Function.Injective f) :
  f 1 = 0 → (Function.invFun f 0) + 1 = 2 := by sorry

end inverse_function_property_l2237_223798


namespace probability_consecutive_is_one_eighteenth_l2237_223724

/-- A standard six-sided die -/
def Die : Type := Fin 6

/-- The set of all possible outcomes when rolling four dice -/
def AllOutcomes : Finset (Die × Die × Die × Die) := sorry

/-- A function to check if four numbers are consecutive -/
def AreConsecutive (a b c d : ℕ) : Prop := sorry

/-- The set of favorable outcomes (four consecutive numbers in any order) -/
def FavorableOutcomes : Finset (Die × Die × Die × Die) := sorry

/-- The probability of rolling four consecutive numbers in any order -/
def ProbabilityConsecutive : ℚ :=
  (FavorableOutcomes.card : ℚ) / (AllOutcomes.card : ℚ)

/-- The main theorem: the probability is 1/18 -/
theorem probability_consecutive_is_one_eighteenth :
  ProbabilityConsecutive = 1 / 18 := by sorry

end probability_consecutive_is_one_eighteenth_l2237_223724


namespace max_n_for_specific_sequence_l2237_223729

/-- Represents an arithmetic sequence with first term a₁, nth term aₙ, and common difference d. -/
structure ArithmeticSequence where
  a₁ : ℤ
  aₙ : ℤ
  d : ℕ+
  n : ℕ
  h_arithmetic : aₙ = a₁ + (n - 1) * d

/-- The maximum value of n for a specific arithmetic sequence. -/
def maxN (seq : ArithmeticSequence) : ℕ :=
  seq.n

/-- Theorem stating the maximum value of n for the given arithmetic sequence. -/
theorem max_n_for_specific_sequence :
  ∀ seq : ArithmeticSequence,
    seq.a₁ = -6 →
    seq.aₙ = 0 →
    seq.n ≥ 3 →
    maxN seq ≤ 7 ∧ ∃ seq' : ArithmeticSequence, seq'.a₁ = -6 ∧ seq'.aₙ = 0 ∧ seq'.n ≥ 3 ∧ maxN seq' = 7 :=
sorry

end max_n_for_specific_sequence_l2237_223729


namespace lcm_is_perfect_square_l2237_223737

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a*b) % (a*b*(a - b)) = 0) : 
  ∃ k : ℕ, Nat.lcm a b = k^2 := by
sorry

end lcm_is_perfect_square_l2237_223737


namespace fruit_seller_apples_l2237_223778

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * (1 - 0.4) = 420 → initial_apples = 700 := by
  sorry

end fruit_seller_apples_l2237_223778


namespace sin_2x_plus_1_equals_shifted_cos_l2237_223763

theorem sin_2x_plus_1_equals_shifted_cos (x : ℝ) : 
  Real.sin (2 * x) + 1 = Real.cos (2 * (x - π / 4)) + 1 := by
  sorry

end sin_2x_plus_1_equals_shifted_cos_l2237_223763


namespace dave_won_fifteen_tickets_l2237_223792

/-- Calculates the number of tickets Dave won later at the arcade -/
def tickets_won_later (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Theorem stating that Dave won 15 tickets later -/
theorem dave_won_fifteen_tickets :
  tickets_won_later 25 22 18 = 15 := by
  sorry

end dave_won_fifteen_tickets_l2237_223792


namespace rectangular_prism_volume_relation_l2237_223716

theorem rectangular_prism_volume_relation (c : ℝ) (hc : c > 0) :
  let a := (4 : ℝ)^(1/3) * c
  let b := (2 : ℝ)^(1/3) * c
  2 * c^3 = a * b * c := by sorry

end rectangular_prism_volume_relation_l2237_223716


namespace existence_of_a_and_b_l2237_223722

theorem existence_of_a_and_b : ∃ (a b : ℝ), a = b + 1 ∧ a^4 = b^4 := by
  sorry

end existence_of_a_and_b_l2237_223722


namespace two_faces_same_edges_l2237_223735

/-- A face of a polyhedron -/
structure Face where
  edges : ℕ
  edges_ge_3 : edges ≥ 3

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Set Face
  nonempty : faces.Nonempty

theorem two_faces_same_edges (P : ConvexPolyhedron) : 
  ∃ f₁ f₂ : Face, f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.edges = f₂.edges :=
sorry

end two_faces_same_edges_l2237_223735


namespace x_zero_value_l2237_223713

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem x_zero_value (x₀ : ℝ) (h : (deriv f) x₀ = 3) :
  x₀ = 1 ∨ x₀ = -1 := by
  sorry

end x_zero_value_l2237_223713


namespace mabel_shark_count_l2237_223704

-- Define the percentage of sharks and other fish
def shark_percentage : ℚ := 25 / 100
def other_fish_percentage : ℚ := 75 / 100

-- Define the number of fish counted on day one
def day_one_count : ℕ := 15

-- Define the multiplier for day two
def day_two_multiplier : ℕ := 3

-- Theorem statement
theorem mabel_shark_count :
  let day_two_count := day_one_count * day_two_multiplier
  let total_fish := day_one_count + day_two_count
  let shark_count := (total_fish : ℚ) * shark_percentage
  shark_count = 15 := by sorry

end mabel_shark_count_l2237_223704


namespace rep_for_A_percent_is_20_l2237_223746

/-- Represents the voting scenario in a city -/
structure VotingScenario where
  total_voters : ℝ
  dem_percent : ℝ
  rep_percent : ℝ
  dem_for_A_percent : ℝ
  total_for_A_percent : ℝ
  rep_for_A_percent : ℝ

/-- The conditions of the voting scenario -/
def city_voting : VotingScenario :=
  { total_voters := 100, -- Assuming 100 for simplicity
    dem_percent := 60,
    rep_percent := 40,
    dem_for_A_percent := 85,
    total_for_A_percent := 59,
    rep_for_A_percent := 20 }

theorem rep_for_A_percent_is_20 (v : VotingScenario) (h1 : v.dem_percent + v.rep_percent = 100) 
    (h2 : v.dem_percent = 60) (h3 : v.dem_for_A_percent = 85) (h4 : v.total_for_A_percent = 59) :
  v.rep_for_A_percent = 20 := by
  sorry

#check rep_for_A_percent_is_20

end rep_for_A_percent_is_20_l2237_223746


namespace expression_simplification_l2237_223743

theorem expression_simplification 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) :
  (a^4 * b^4 + a^4 * c^4 + b^4 * c^4) / 
  ((a^2 - b*c)^2 * (b^2 - a*c)^2 * (c^2 - a*b)^2) = 
  1 / (a^2 - b*c)^2 :=
by sorry

end expression_simplification_l2237_223743


namespace light_ray_reflection_l2237_223744

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The starting point A -/
def A : Point := ⟨-3, 4⟩

/-- The final point B -/
def B : Point := ⟨-2, 6⟩

/-- The equation of the light ray after reflecting off the y-axis -/
def final_ray : Line := ⟨2, 1, -2⟩

/-- Theorem stating that the final ray passes through point B and has the correct equation -/
theorem light_ray_reflection :
  (B.on_line final_ray) ∧
  (final_ray.a = 2 ∧ final_ray.b = 1 ∧ final_ray.c = -2) := by sorry

end light_ray_reflection_l2237_223744


namespace carla_chicken_farm_problem_l2237_223747

/-- The percentage of chickens that died in Carla's farm -/
def percentage_died (initial_chickens final_chickens : ℕ) : ℚ :=
  let bought_chickens := final_chickens - initial_chickens
  let died_chickens := bought_chickens / 10
  (died_chickens : ℚ) / initial_chickens * 100

theorem carla_chicken_farm_problem :
  percentage_died 400 1840 = 40 := by
  sorry

end carla_chicken_farm_problem_l2237_223747


namespace remainder_sum_l2237_223766

theorem remainder_sum (a b : ℤ) 
  (ha : a % 60 = 53) 
  (hb : b % 45 = 22) : 
  (a + b) % 30 = 15 := by
sorry

end remainder_sum_l2237_223766


namespace tile_arrangements_l2237_223750

def brown_tiles : ℕ := 2
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 4

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem tile_arrangements :
  (Nat.factorial total_tiles) / 
  (Nat.factorial yellow_tiles * Nat.factorial green_tiles * 
   Nat.factorial brown_tiles * Nat.factorial purple_tiles) = 12600 := by
  sorry

end tile_arrangements_l2237_223750


namespace cookies_per_bag_l2237_223721

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : total_cookies = 14)
  (h2 : num_bags = 7)
  (h3 : total_cookies = num_bags * cookies_per_bag) :
  cookies_per_bag = 2 := by
  sorry

end cookies_per_bag_l2237_223721


namespace valid_basis_vectors_l2237_223797

def vector_a : Fin 2 → ℝ := ![3, 4]

def vector_e1 : Fin 2 → ℝ := ![-1, 2]
def vector_e2 : Fin 2 → ℝ := ![3, -1]

theorem valid_basis_vectors :
  ∃ (x y : ℝ), vector_a = x • vector_e1 + y • vector_e2 ∧
  ¬(∃ (k : ℝ), vector_e1 = k • vector_e2) :=
by sorry

end valid_basis_vectors_l2237_223797


namespace chessboard_inner_square_probability_l2237_223710

/-- Represents a square chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Calculates the total number of squares on the chessboard -/
def total_squares (board : Chessboard) : ℕ :=
  board.size * board.size

/-- Calculates the number of squares in the outermost two rows and columns -/
def outer_squares (board : Chessboard) : ℕ :=
  4 * board.size - 4

/-- Calculates the number of inner squares not touching the outermost two rows or columns -/
def inner_squares (board : Chessboard) : ℕ :=
  total_squares board - outer_squares board

/-- The probability of choosing an inner square -/
def inner_square_probability (board : Chessboard) : ℚ :=
  inner_squares board / total_squares board

theorem chessboard_inner_square_probability :
  ∃ (board : Chessboard), board.size = 10 ∧ inner_square_probability board = 17 / 25 := by
  sorry

end chessboard_inner_square_probability_l2237_223710


namespace range_of_a_l2237_223772

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x₀ : ℝ, x₀^2 + 2*x₀ + a > 0)
  (h2 : ∀ x : ℝ, x > 0 → x + 1/x > a) : 
  1 < a ∧ a < 2 := by
sorry

end range_of_a_l2237_223772


namespace city_inhabitants_problem_l2237_223734

theorem city_inhabitants_problem :
  ∃ n : ℕ,
    n > 150 ∧
    (∃ x : ℕ, n = x^2) ∧
    (∃ y : ℕ, n + 1000 = y^2 + 1) ∧
    (∃ z : ℕ, n + 2000 = z^2) ∧
    n = 249001 := by
  sorry

end city_inhabitants_problem_l2237_223734


namespace product_of_four_numbers_l2237_223728

theorem product_of_four_numbers (A B C D : ℝ) : 
  A > 0 → B > 0 → C > 0 → D > 0 →
  A + B + C + D = 40 →
  A + 3 = B - 3 ∧ A + 3 = C * 3 ∧ A + 3 = D / 3 →
  A * B * C * D = 2666.25 := by
sorry

end product_of_four_numbers_l2237_223728


namespace cryptarithm_solution_exists_l2237_223760

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_different_digits (Φ E B P A J : ℕ) : Prop :=
  is_valid_digit Φ ∧ is_valid_digit E ∧ is_valid_digit B ∧ 
  is_valid_digit P ∧ is_valid_digit A ∧ is_valid_digit J ∧
  Φ ≠ E ∧ Φ ≠ B ∧ Φ ≠ P ∧ Φ ≠ A ∧ Φ ≠ J ∧
  E ≠ B ∧ E ≠ P ∧ E ≠ A ∧ E ≠ J ∧
  B ≠ P ∧ B ≠ A ∧ B ≠ J ∧
  P ≠ A ∧ P ≠ J ∧
  A ≠ J

theorem cryptarithm_solution_exists :
  ∃ (Φ E B P A J : ℕ), 
    are_different_digits Φ E B P A J ∧
    (Φ : ℚ) / E + (B * 10 + P : ℚ) / A / J = 1 :=
sorry

end cryptarithm_solution_exists_l2237_223760


namespace triangle_problem_l2237_223780

open Real

theorem triangle_problem (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * sin (A - C) = sin B) :
  sin A = (3 * sqrt 10) / 10 ∧
  (∀ (AB : ℝ), AB = 5 → ∃ (h : ℝ), h = 6 ∧ h * AB / 2 = sin C * (AB * sin A / sin C) * (AB * sin B / sin C) / 2) :=
sorry

end triangle_problem_l2237_223780


namespace mindys_tax_rate_mindys_tax_rate_is_15_percent_l2237_223701

/-- Calculates Mindy's tax rate given the conditions of the problem -/
theorem mindys_tax_rate (morks_income : ℝ) (morks_tax_rate : ℝ) (mindys_income_multiplier : ℝ) (combined_tax_rate : ℝ) : ℝ :=
  let mindys_income := mindys_income_multiplier * morks_income
  let total_income := morks_income + mindys_income
  let mindys_tax_rate := (combined_tax_rate * total_income - morks_tax_rate * morks_income) / mindys_income
  mindys_tax_rate

/-- Proves that Mindy's tax rate is 15% given the problem conditions -/
theorem mindys_tax_rate_is_15_percent :
  mindys_tax_rate 1 0.45 4 0.21 = 0.15 := by
  sorry

end mindys_tax_rate_mindys_tax_rate_is_15_percent_l2237_223701


namespace quadrilaterals_with_fixed_point_l2237_223720

theorem quadrilaterals_with_fixed_point (n : ℕ) (k : ℕ) :
  n = 11 ∧ k = 3 → Nat.choose n k = 165 := by sorry

end quadrilaterals_with_fixed_point_l2237_223720


namespace fourth_power_of_cube_root_l2237_223740

theorem fourth_power_of_cube_root (x : ℝ) : 
  x = (3 + Real.sqrt (1 + Real.sqrt 5)) ^ (1/3) → x^4 = 9 + 12 * Real.sqrt 6 := by
  sorry

end fourth_power_of_cube_root_l2237_223740


namespace probability_inequality_l2237_223739

theorem probability_inequality (p q : ℝ) (m n : ℕ+) 
  (h1 : p ≥ 0) (h2 : q ≥ 0) (h3 : p + q = 1) :
  (1 - p ^ (m : ℝ)) ^ (n : ℝ) + (1 - q ^ (n : ℝ)) ^ (m : ℝ) ≥ 1 := by
  sorry

end probability_inequality_l2237_223739


namespace keiths_cds_l2237_223703

/-- Calculates the number of CDs Keith wanted to buy based on his total spending and the price per CD -/
theorem keiths_cds (speakers_cost cd_player_cost tires_cost total_spent cd_price : ℝ) :
  speakers_cost = 136.01 →
  cd_player_cost = 139.38 →
  tires_cost = 112.46 →
  total_spent = 387.85 →
  cd_price = 6.16 →
  speakers_cost + cd_player_cost + tires_cost = total_spent →
  ⌊total_spent / cd_price⌋ = 62 :=
by sorry

end keiths_cds_l2237_223703


namespace cubic_odd_and_increasing_l2237_223776

def f (x : ℝ) : ℝ := x^3

theorem cubic_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end cubic_odd_and_increasing_l2237_223776


namespace files_per_folder_l2237_223771

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) :
  initial_files = 93 →
  deleted_files = 21 →
  num_folders = 9 →
  (initial_files - deleted_files) / num_folders = 8 :=
by
  sorry

end files_per_folder_l2237_223771


namespace guppies_needed_per_day_l2237_223723

/-- The number of guppies eaten by a moray eel per day -/
def moray_eel_guppies : ℕ := 20

/-- The number of betta fish -/
def num_betta_fish : ℕ := 5

/-- The number of guppies eaten by each betta fish per day -/
def betta_fish_guppies : ℕ := 7

/-- The total number of guppies needed per day -/
def total_guppies : ℕ := moray_eel_guppies + num_betta_fish * betta_fish_guppies

theorem guppies_needed_per_day : total_guppies = 55 := by
  sorry

end guppies_needed_per_day_l2237_223723


namespace x_value_l2237_223738

theorem x_value (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end x_value_l2237_223738


namespace pyramid_volume_change_specific_pyramid_volume_l2237_223707

/-- Given a pyramid with a triangular base and initial volume V, 
    if the base height is doubled, base dimensions are tripled, 
    and the pyramid's height is increased by 40%, 
    then the new volume is 8.4 * V. -/
theorem pyramid_volume_change (V : ℝ) : 
  V > 0 → 
  (2 * 3 * 3 * 1.4) * V = 8.4 * V :=
by sorry

/-- The new volume of the specific pyramid is 604.8 cubic inches. -/
theorem specific_pyramid_volume : 
  (8.4 : ℝ) * 72 = 604.8 :=
by sorry

end pyramid_volume_change_specific_pyramid_volume_l2237_223707


namespace product_equals_442_l2237_223753

/-- Converts a list of digits in a given base to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

/-- The binary representation of the first number -/
def binary_num : List Nat := [1, 0, 1, 1]

/-- The ternary representation of the second number -/
def ternary_num : List Nat := [1, 2, 0, 1]

theorem product_equals_442 :
  (to_decimal binary_num 2) * (to_decimal ternary_num 3) = 442 := by
  sorry

end product_equals_442_l2237_223753


namespace total_monthly_cost_l2237_223714

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the storage details -/
structure StorageDetails where
  boxDim : BoxDimensions
  totalVolume : ℝ
  costPerBox : ℝ

/-- Theorem stating that the total monthly cost for record storage is $480 -/
theorem total_monthly_cost (s : StorageDetails)
  (h1 : s.boxDim = ⟨15, 12, 10⟩)
  (h2 : s.totalVolume = 1080000)
  (h3 : s.costPerBox = 0.8) :
  (s.totalVolume / boxVolume s.boxDim) * s.costPerBox = 480 := by
  sorry


end total_monthly_cost_l2237_223714


namespace inequality_proof_l2237_223727

theorem inequality_proof (x : ℝ) (h : x > 0) : Real.log (Real.exp 2 / x) ≤ (1 + x) / x := by
  sorry

end inequality_proof_l2237_223727


namespace share_division_l2237_223787

theorem share_division (total : ℚ) (a b c : ℚ) : 
  total = 700 →
  a + b + c = total →
  a = b / 2 →
  b = c / 2 →
  c = 400 := by
sorry

end share_division_l2237_223787


namespace average_visitors_is_276_l2237_223758

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays := 4
  let totalOtherDays := 26
  let totalVisitors := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

/-- Theorem stating that the average number of visitors is 276 given the specified conditions -/
theorem average_visitors_is_276 :
  averageVisitors 510 240 = 276 := by
  sorry

end average_visitors_is_276_l2237_223758


namespace min_value_and_reciprocal_sum_l2237_223757

noncomputable def f (a b c x : ℝ) : ℝ := |x + a| + |x - b| + c

theorem min_value_and_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, f a b c x ≥ 5) 
  (hex : ∃ x, f a b c x = 5) :
  (a + b + c = 5) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → 1/a' + 1/b' + 1/c' ≥ 9/5) ∧
  (∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 1/a' + 1/b' + 1/c' = 9/5) :=
by sorry

end min_value_and_reciprocal_sum_l2237_223757


namespace angle5_measure_l2237_223773

-- Define the angles
variable (angle1 angle2 angle5 : ℝ)

-- Define the conditions
axiom angle1_fraction : angle1 = (1/4) * angle2
axiom supplementary : angle2 + angle5 = 180

-- State the theorem
theorem angle5_measure : angle5 = 36 := by
  sorry

end angle5_measure_l2237_223773


namespace total_carrots_l2237_223769

theorem total_carrots (sally_carrots fred_carrots : ℕ) :
  sally_carrots = 6 → fred_carrots = 4 → sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l2237_223769


namespace last_ten_seconds_distance_l2237_223726

/-- The distance function of a plane's taxiing after landing -/
def distance (t : ℝ) : ℝ := 60 * t - 1.5 * t^2

/-- The time at which the plane stops -/
def stop_time : ℝ := 20

/-- Theorem: The plane travels 150 meters in the last 10 seconds before stopping -/
theorem last_ten_seconds_distance : 
  distance stop_time - distance (stop_time - 10) = 150 := by
  sorry

end last_ten_seconds_distance_l2237_223726


namespace parallel_lines_a_value_l2237_223790

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The value of 'a' for which the given lines are parallel -/
theorem parallel_lines_a_value :
  (∀ x y : ℝ, 3 * y + 6 * a = 9 * x ↔ y - 2 = (2 * a - 3) * x) → a = 3 := by
  sorry

#check parallel_lines_a_value

end parallel_lines_a_value_l2237_223790


namespace gift_contribution_max_l2237_223733

/-- Given a group of people contributing money, calculates the maximum possible contribution by a single person. -/
def max_contribution (n : ℕ) (total : ℚ) (min_contribution : ℚ) : ℚ :=
  total - (n - 1 : ℚ) * min_contribution

/-- Theorem stating the maximum possible contribution in the given scenario. -/
theorem gift_contribution_max (n : ℕ) (total : ℚ) (min_contribution : ℚ)
  (h_n : n = 10)
  (h_total : total = 20)
  (h_min : min_contribution = 1)
  (h_positive : ∀ i, i ≤ n → min_contribution ≤ (max_contribution n total min_contribution)) :
  max_contribution n total min_contribution = 11 :=
by sorry

end gift_contribution_max_l2237_223733


namespace quadratic_factorization_l2237_223762

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end quadratic_factorization_l2237_223762
