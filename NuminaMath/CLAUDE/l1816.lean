import Mathlib

namespace min_value_neg_half_l1816_181605

/-- A function f with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

/-- The maximum value of f on (0, +∞) -/
def max_value : ℝ := 5

/-- Theorem: The minimum value of f on (-∞, 0) is -1 -/
theorem min_value_neg_half (a b : ℝ) :
  (∀ x > 0, f a b x ≤ max_value) →
  (∃ x > 0, f a b x = max_value) →
  (∀ x < 0, f a b x ≥ -1) ∧
  (∃ x < 0, f a b x = -1) :=
sorry

end min_value_neg_half_l1816_181605


namespace students_making_stars_l1816_181629

theorem students_making_stars (total_stars : ℕ) (stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) :
  total_stars / stars_per_student = 124 := by
  sorry

end students_making_stars_l1816_181629


namespace flour_recipe_reduction_reduced_recipe_as_mixed_number_l1816_181649

theorem flour_recipe_reduction :
  let original_recipe : ℚ := 19/4  -- 4 3/4 as an improper fraction
  let reduced_recipe : ℚ := original_recipe / 3
  reduced_recipe = 19/12 := by sorry

theorem reduced_recipe_as_mixed_number :
  (19 : ℚ) / 12 = 1 + 7/12 := by sorry

end flour_recipe_reduction_reduced_recipe_as_mixed_number_l1816_181649


namespace rectangle_perimeter_l1816_181662

/-- Given a large square with side length 8y and a smaller central square with side length 3y,
    where the large square is divided into the smaller central square and four congruent rectangles,
    the perimeter of one of these rectangles is 16y. -/
theorem rectangle_perimeter (y : ℝ) : 
  let large_square_side : ℝ := 8 * y
  let small_square_side : ℝ := 3 * y
  let rectangle_width : ℝ := small_square_side
  let rectangle_height : ℝ := large_square_side - small_square_side
  let rectangle_perimeter : ℝ := 2 * (rectangle_width + rectangle_height)
  rectangle_perimeter = 16 * y :=
by sorry

end rectangle_perimeter_l1816_181662


namespace smallest_angle_measure_l1816_181639

-- Define a right triangle with acute angles in the ratio 3:2
structure RightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  right_angle : ℝ
  is_right_triangle : right_angle = 90
  acute_angle_sum : angle1 + angle2 = 90
  angle_ratio : angle1 / angle2 = 3 / 2

-- Theorem statement
theorem smallest_angle_measure (t : RightTriangle) : 
  min t.angle1 t.angle2 = 36 := by
  sorry

end smallest_angle_measure_l1816_181639


namespace not_perfect_square_sum_of_squares_l1816_181670

theorem not_perfect_square_sum_of_squares (x y : ℤ) :
  ¬ ∃ (n : ℤ), (x^2 + x + 1)^2 + (y^2 + y + 1)^2 = n^2 := by
sorry

end not_perfect_square_sum_of_squares_l1816_181670


namespace divides_m_implies_divides_m_times_n_plus_one_l1816_181661

theorem divides_m_implies_divides_m_times_n_plus_one (m n : ℤ) :
  n ∣ m * (n + 1) → n ∣ m := by
  sorry

end divides_m_implies_divides_m_times_n_plus_one_l1816_181661


namespace tangent_and_mean_value_theorem_l1816_181658

noncomputable section

/-- The function f(x) = x^2 + a(x + ln x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * (x + Real.log x)

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x + a * (1 + 1/x)

theorem tangent_and_mean_value_theorem (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f_deriv a x₀ = 2*(a+1) ∧ f a x₀ = (a+1)*(2*x₀-1) - a - 1) ∧
  (∃ ξ : ℝ, 1 < ξ ∧ ξ < Real.exp 1 ∧ f_deriv a ξ = (f a (Real.exp 1) - f a 1) / (Real.exp 1 - 1)) :=
sorry

end

end tangent_and_mean_value_theorem_l1816_181658


namespace problem_statement_l1816_181643

theorem problem_statement (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end problem_statement_l1816_181643


namespace bonus_difference_l1816_181651

/-- Prove that given a total bonus of $5,000 divided between two employees,
    where the senior employee receives $1,900 and the junior employee receives $3,100,
    the difference between the junior employee's bonus and the senior employee's bonus is $1,200. -/
theorem bonus_difference (total_bonus senior_bonus junior_bonus : ℕ) : 
  total_bonus = 5000 →
  senior_bonus = 1900 →
  junior_bonus = 3100 →
  junior_bonus - senior_bonus = 1200 := by
  sorry

end bonus_difference_l1816_181651


namespace heavy_washes_count_l1816_181684

/-- Represents the number of gallons of water used for different wash types and conditions --/
structure WashingMachine where
  heavyWashWater : ℕ
  regularWashWater : ℕ
  lightWashWater : ℕ
  bleachRinseWater : ℕ
  regularWashCount : ℕ
  lightWashCount : ℕ
  bleachedLoadsCount : ℕ
  totalWaterUsage : ℕ

/-- Calculates the number of heavy washes given the washing machine parameters --/
def calculateHeavyWashes (wm : WashingMachine) : ℕ :=
  (wm.totalWaterUsage - 
   (wm.regularWashWater * wm.regularWashCount + 
    wm.lightWashWater * wm.lightWashCount + 
    wm.bleachRinseWater * wm.bleachedLoadsCount)) / wm.heavyWashWater

/-- Theorem stating that the number of heavy washes is 2 given the specific conditions --/
theorem heavy_washes_count (wm : WashingMachine) 
  (h1 : wm.heavyWashWater = 20)
  (h2 : wm.regularWashWater = 10)
  (h3 : wm.lightWashWater = 2)
  (h4 : wm.bleachRinseWater = 2)
  (h5 : wm.regularWashCount = 3)
  (h6 : wm.lightWashCount = 1)
  (h7 : wm.bleachedLoadsCount = 2)
  (h8 : wm.totalWaterUsage = 76) :
  calculateHeavyWashes wm = 2 := by
  sorry

#eval calculateHeavyWashes {
  heavyWashWater := 20,
  regularWashWater := 10,
  lightWashWater := 2,
  bleachRinseWater := 2,
  regularWashCount := 3,
  lightWashCount := 1,
  bleachedLoadsCount := 2,
  totalWaterUsage := 76
}

end heavy_washes_count_l1816_181684


namespace bus_average_speed_l1816_181620

/-- Proves that the average speed of a bus line is 60 km/h given specific conditions -/
theorem bus_average_speed
  (stop_interval : ℕ) -- Time interval between stops in minutes
  (num_stops : ℕ) -- Number of stops to the destination
  (distance : ℝ) -- Distance to the destination in kilometers
  (h1 : stop_interval = 5)
  (h2 : num_stops = 8)
  (h3 : distance = 40) :
  distance / (↑(stop_interval * num_stops) / 60) = 60 :=
by sorry

end bus_average_speed_l1816_181620


namespace triangle_properties_l1816_181641

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c) 
  (h2 : t.c = 3 * t.a) : 
  t.B = π/3 ∧ Real.sin t.A = Real.sqrt 21 / 14 := by
  sorry

end triangle_properties_l1816_181641


namespace birdhouse_charge_for_two_l1816_181656

/-- The cost of a birdhouse for Denver -/
def birdhouse_cost (wood_pieces : ℕ) (wood_price paint_cost labor_cost : ℚ) : ℚ :=
  wood_pieces * wood_price + paint_cost + labor_cost

/-- The selling price of a birdhouse -/
def birdhouse_price (cost profit : ℚ) : ℚ :=
  cost + profit

/-- The total charge for multiple birdhouses -/
def total_charge (price : ℚ) (quantity : ℕ) : ℚ :=
  price * quantity

theorem birdhouse_charge_for_two :
  let wood_pieces : ℕ := 7
  let wood_price : ℚ := 3/2  -- $1.50
  let paint_cost : ℚ := 3
  let labor_cost : ℚ := 9/2  -- $4.50
  let profit : ℚ := 11/2  -- $5.50
  let cost := birdhouse_cost wood_pieces wood_price paint_cost labor_cost
  let price := birdhouse_price cost profit
  let quantity : ℕ := 2
  total_charge price quantity = 47
  := by sorry

end birdhouse_charge_for_two_l1816_181656


namespace cheaper_module_cost_l1816_181686

theorem cheaper_module_cost (expensive_cost : ℝ) (total_modules : ℕ) (cheap_modules : ℕ) (total_value : ℝ) :
  expensive_cost = 10 →
  total_modules = 11 →
  cheap_modules = 10 →
  total_value = 45 →
  ∃ (cheap_cost : ℝ), cheap_cost * cheap_modules + expensive_cost * (total_modules - cheap_modules) = total_value ∧ cheap_cost = 3.5 := by
  sorry

end cheaper_module_cost_l1816_181686


namespace proposition_p_and_not_q_l1816_181682

theorem proposition_p_and_not_q : 
  (∃ x : ℝ, x^2 - x + 1 ≥ 0) ∧ 
  (∃ a b : ℝ, a^2 < b^2 ∧ a ≥ b) := by
  sorry

end proposition_p_and_not_q_l1816_181682


namespace ms_hatcher_students_l1816_181688

theorem ms_hatcher_students (third_graders : ℕ) (fourth_graders : ℕ) (fifth_graders : ℕ) : 
  third_graders = 20 →
  fourth_graders = 2 * third_graders →
  fifth_graders = third_graders / 2 →
  third_graders + fourth_graders + fifth_graders = 70 := by
  sorry

end ms_hatcher_students_l1816_181688


namespace bob_total_candies_l1816_181618

/-- Calculates Bob's share of candies given the total amount and the ratio --/
def bobShare (total : ℕ) (samRatio : ℕ) (bobRatio : ℕ) : ℕ :=
  (total * bobRatio) / (samRatio + bobRatio)

/-- Theorem: Bob receives 64 candies in total --/
theorem bob_total_candies : 
  let chewingGums := bobShare 45 2 3
  let chocolateBars := bobShare 60 3 1
  let assortedCandies := 45 / 2
  chewingGums + chocolateBars + assortedCandies = 64 := by
  sorry

#eval bobShare 45 2 3 -- Should output 27
#eval bobShare 60 3 1 -- Should output 15
#eval 45 / 2          -- Should output 22

end bob_total_candies_l1816_181618


namespace incorrect_statement_l1816_181657

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the containment relation for lines and planes
variable (contains : Plane → Line → Prop)

-- State the theorem
theorem incorrect_statement 
  (α β : Plane) (m n : Line) : 
  ¬(∀ α β m n, parallelLinePlane m α ∧ intersect α β = n → parallelLine m n) := by
  sorry

end incorrect_statement_l1816_181657


namespace max_positive_integers_l1816_181602

theorem max_positive_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  ∃ (pos : Finset ℤ), pos ⊆ {a, b, c, d, e, f} ∧ pos.card ≤ 5 ∧
  (∀ x ∈ pos, x > 0) ∧
  (∀ pos' : Finset ℤ, pos' ⊆ {a, b, c, d, e, f} → (∀ x ∈ pos', x > 0) → pos'.card ≤ pos.card) :=
by sorry

end max_positive_integers_l1816_181602


namespace quadratic_inequality_equivalence_l1816_181667

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_equivalence_l1816_181667


namespace second_to_first_angle_ratio_l1816_181627

def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

theorem second_to_first_angle_ratio 
  (a b c : ℝ) 
  (h_triangle : Triangle a b c)
  (h_second_multiple : ∃ k : ℝ, b = k * a)
  (h_third : c = 2 * a - 12)
  (h_measures : a = 32 ∧ b = 96 ∧ c = 52) :
  b / a = 3 := by
sorry

end second_to_first_angle_ratio_l1816_181627


namespace kenny_trumpet_practice_l1816_181606

def basketball_hours : ℕ := 10

def running_hours (b : ℕ) : ℕ := 2 * b

def trumpet_hours (r : ℕ) : ℕ := 2 * r

def total_practice_hours (b r t : ℕ) : ℕ := b + r + t

theorem kenny_trumpet_practice (x y : ℕ) :
  let b := basketball_hours
  let r := running_hours b
  let t := trumpet_hours r
  total_practice_hours b r t = x + y →
  t = 40 := by
sorry

end kenny_trumpet_practice_l1816_181606


namespace probability_D_given_E_l1816_181611

-- Define the regions D and E
def region_D (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ x + 1 ≤ y ∧ y ≤ 2

def region_E (x y : ℝ) : Prop :=
  -1 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2

-- Define the area function
def area (region : ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem probability_D_given_E :
  (area region_D) / (area region_E) = 3/8 := by sorry

end probability_D_given_E_l1816_181611


namespace cube_prism_cuboid_rectangular_prism_subset_l1816_181622

-- Define the sets
variable (M : Set (Set ℝ)) -- Set of all right prisms
variable (N : Set (Set ℝ)) -- Set of all cuboids
variable (Q : Set (Set ℝ)) -- Set of all cubes
variable (P : Set (Set ℝ)) -- Set of all right rectangular prisms

-- State the theorem
theorem cube_prism_cuboid_rectangular_prism_subset : Q ⊆ M ∧ M ⊆ N ∧ N ⊆ P := by
  sorry

end cube_prism_cuboid_rectangular_prism_subset_l1816_181622


namespace log_equality_implies_y_value_l1816_181637

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := sorry

-- Define the variables
variable (a b c x : ℝ)
variable (p q r y : ℝ)

-- State the theorem
theorem log_equality_implies_y_value
  (h1 : log a / p = log b / q)
  (h2 : log b / q = log c / r)
  (h3 : log c / r = log x)
  (h4 : x ≠ 1)
  (h5 : b^3 / (a^2 * c) = x^y) :
  y = 3*q - 2*p - r := by
  sorry

end log_equality_implies_y_value_l1816_181637


namespace coconut_grove_yield_is_120_l1816_181681

/-- Represents the yield of x trees in a coconut grove with specific conditions -/
def coconut_grove_yield (x : ℕ) (yield_x : ℕ) : Prop :=
  let yield_xplus2 : ℕ := 30 * (x + 2)
  let yield_xminus2 : ℕ := 180 * (x - 2)
  let total_trees : ℕ := (x + 2) + x + (x - 2)
  let total_yield : ℕ := yield_xplus2 + (x * yield_x) + yield_xminus2
  (total_yield = total_trees * 100) ∧ (x = 10)

theorem coconut_grove_yield_is_120 :
  coconut_grove_yield 10 120 := by sorry

end coconut_grove_yield_is_120_l1816_181681


namespace peanuts_added_l1816_181633

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) : 
  initial_peanuts = 4 →
  final_peanuts = 16 →
  final_peanuts - initial_peanuts = 12 :=
by
  sorry

end peanuts_added_l1816_181633


namespace work_completion_time_l1816_181660

/-- The time it takes for A, B, and C to complete a work together -/
def time_together (time_A time_B time_C : ℚ) : ℚ :=
  1 / (1 / time_A + 1 / time_B + 1 / time_C)

/-- Theorem stating that A, B, and C can complete the work in 2 days -/
theorem work_completion_time :
  time_together 4 10 (20 / 3) = 2 := by sorry

end work_completion_time_l1816_181660


namespace geometric_sequence_inequality_l1816_181600

/-- Given a geometric sequence {a_n}, prove that a_1^2 + a_3^2 ≥ 2a_2^2 -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) 
  (h : ∀ n, a (n + 1) = a n * q) : 
  a 1^2 + a 3^2 ≥ 2 * a 2^2 := by
  sorry

end geometric_sequence_inequality_l1816_181600


namespace multiples_5_or_7_not_both_main_theorem_l1816_181698

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_5_or_7_not_both (upper_bound : ℕ) 
  (h_upper_bound : upper_bound = 101) : ℕ := by
  let multiples_5 := count_multiples upper_bound 5
  let multiples_7 := count_multiples upper_bound 7
  let multiples_35 := count_multiples upper_bound 35
  exact (multiples_5 + multiples_7 - 2 * multiples_35)

theorem main_theorem : multiples_5_or_7_not_both 101 rfl = 30 := by
  sorry

end multiples_5_or_7_not_both_main_theorem_l1816_181698


namespace inequality_one_inequality_two_l1816_181648

-- Part 1
theorem inequality_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3 := by sorry

-- Part 2
theorem inequality_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  a * b + b * c + a * c ≤ 1 / 3 := by sorry

end inequality_one_inequality_two_l1816_181648


namespace wire_service_reporters_l1816_181690

theorem wire_service_reporters (total : ℝ) (local_politics : ℝ) (politics : ℝ) : 
  local_politics = 0.2 * total →
  local_politics = 0.8 * politics →
  (total - politics) / total = 0.75 :=
by sorry

end wire_service_reporters_l1816_181690


namespace probability_multiple_2_3_7_l1816_181659

/-- The number of integers from 1 to n that are divisible by at least one of a, b, or c -/
def countMultiples (n : ℕ) (a b c : ℕ) : ℕ :=
  (n / a + n / b + n / c) - (n / lcm a b + n / lcm a c + n / lcm b c) + n / lcm a (lcm b c)

/-- The probability of selecting a multiple of 2, 3, or 7 from the first 150 positive integers -/
theorem probability_multiple_2_3_7 : 
  countMultiples 150 2 3 7 = 107 := by sorry

end probability_multiple_2_3_7_l1816_181659


namespace largest_integer_difference_in_triangle_l1816_181678

theorem largest_integer_difference_in_triangle (n : ℕ) (hn : n ≥ 4) :
  (∃ k : ℕ, k > 0 ∧
    (∀ k' : ℕ, k' > k →
      ¬∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧
        c - b ≥ k' ∧ b - a ≥ k' ∧ a + b ≥ c + 1) ∧
    (∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧
      c - b ≥ k ∧ b - a ≥ k ∧ a + b ≥ c + 1)) ∧
  (∀ k : ℕ, (∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n ∧
    c - b ≥ k ∧ b - a ≥ k ∧ a + b ≥ c + 1) →
    k ≤ (n - 1) / 3) :=
by sorry

end largest_integer_difference_in_triangle_l1816_181678


namespace skew_lines_distance_l1816_181691

/-- Given two skew lines a and b forming an angle θ, with their common perpendicular AA' of length d,
    and points E on a and F on b such that A'E = m and AF = n, the distance EF is given by
    √(d² + m² + n² ± 2mn cos θ). -/
theorem skew_lines_distance (θ d m n : ℝ) : ∃ (EF : ℝ), 
  EF = Real.sqrt (d^2 + m^2 + n^2 + 2*m*n*(Real.cos θ)) ∨
  EF = Real.sqrt (d^2 + m^2 + n^2 - 2*m*n*(Real.cos θ)) :=
by sorry


end skew_lines_distance_l1816_181691


namespace product_of_roots_l1816_181677

theorem product_of_roots (x : ℝ) : 
  (x^3 - 9*x^2 + 27*x - 8 = 0) → 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 8 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 8) :=
by sorry

end product_of_roots_l1816_181677


namespace percentage_problem_l1816_181617

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 16 = 40 → P = 250 := by
  sorry

end percentage_problem_l1816_181617


namespace cake_serving_capacity_l1816_181669

-- Define the original cake properties
def original_radius : ℝ := 20
def original_people_served : ℕ := 4

-- Define the new cake radius
def new_radius : ℝ := 50

-- Theorem statement
theorem cake_serving_capacity :
  ∃ (new_people_served : ℕ), 
    new_people_served = 25 ∧
    (new_radius^2 / original_radius^2) * original_people_served = new_people_served :=
by
  sorry

end cake_serving_capacity_l1816_181669


namespace median_length_half_side_l1816_181612

/-- Prove that the length of a median in a triangle is half the length of its corresponding side. -/
theorem median_length_half_side {A B C : ℝ × ℝ} : 
  let M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)  -- Midpoint of BC
  dist A M = (1/2) * dist B C := by
  sorry

end median_length_half_side_l1816_181612


namespace quadratic_polynomial_property_l1816_181663

theorem quadratic_polynomial_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h_distinct_a_b : a ≠ b) (h_distinct_b_c : b ≠ c) (h_distinct_a_c : a ≠ c)
  (h_quadratic : ∃ p q r : ℝ, ∀ x, f x = p * x^2 + q * x + r)
  (h_f_a : f a = b * c) (h_f_b : f b = c * a) (h_f_c : f c = a * b) :
  f (a + b + c) = a * b + b * c + a * c := by
  sorry

end quadratic_polynomial_property_l1816_181663


namespace parabola_equation_l1816_181674

/-- A parabola with focus at (-2, 0) has the standard equation y^2 = -8x -/
theorem parabola_equation (F : ℝ × ℝ) (h : F = (-2, 0)) : 
  ∃ (x y : ℝ), y^2 = -8*x := by sorry

end parabola_equation_l1816_181674


namespace pizza_median_theorem_l1816_181626

/-- Represents the pizza sales data for a day -/
structure PizzaSalesData where
  total_slices : ℕ
  total_customers : ℕ
  min_slices_per_customer : ℕ

/-- Calculates the maximum possible median number of slices per customer -/
def max_possible_median (data : PizzaSalesData) : ℚ :=
  sorry

/-- Theorem stating the maximum possible median for the given scenario -/
theorem pizza_median_theorem (data : PizzaSalesData) 
  (h1 : data.total_slices = 310)
  (h2 : data.total_customers = 150)
  (h3 : data.min_slices_per_customer = 1) :
  max_possible_median data = 7 := by
  sorry

end pizza_median_theorem_l1816_181626


namespace jackets_sold_after_noon_l1816_181697

theorem jackets_sold_after_noon :
  let total_jackets : ℕ := 214
  let price_before_noon : ℚ := 31.95
  let price_after_noon : ℚ := 18.95
  let total_receipts : ℚ := 5108.30
  let jackets_after_noon : ℕ := 133
  let jackets_before_noon : ℕ := total_jackets - jackets_after_noon
  (jackets_before_noon : ℚ) * price_before_noon + (jackets_after_noon : ℚ) * price_after_noon = total_receipts →
  jackets_after_noon = 133 :=
by
  sorry

end jackets_sold_after_noon_l1816_181697


namespace function_range_l1816_181647

/-- Given a function f(x) = x³ - 3a²x + a where a > 0, 
    if its maximum value is positive and its minimum value is negative, 
    then a > √2/2 -/
theorem function_range (a : ℝ) (h1 : a > 0) 
  (f : ℝ → ℝ) (h2 : ∀ x, f x = x^3 - 3*a^2*x + a) 
  (h3 : ∃ M, ∀ x, f x ≤ M ∧ M > 0)  -- maximum value is positive
  (h4 : ∃ m, ∀ x, f x ≥ m ∧ m < 0)  -- minimum value is negative
  : a > Real.sqrt 2 / 2 := by
  sorry

end function_range_l1816_181647


namespace puppy_cost_problem_l1816_181680

theorem puppy_cost_problem (total_cost : ℕ) (sale_price : ℕ) (distinct_cost1 : ℕ) (distinct_cost2 : ℕ) :
  total_cost = 2200 →
  sale_price = 180 →
  distinct_cost1 = 250 →
  distinct_cost2 = 300 →
  ∃ (remaining_price : ℕ),
    4 * sale_price + distinct_cost1 + distinct_cost2 + 2 * remaining_price = total_cost ∧
    remaining_price = 465 := by
  sorry

end puppy_cost_problem_l1816_181680


namespace rectangular_solid_surface_area_l1816_181699

/-- A rectangular solid with prime edge lengths and volume 143 has surface area 382 -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 143 →
  2 * (a * b + b * c + c * a) = 382 := by
  sorry

end rectangular_solid_surface_area_l1816_181699


namespace y_in_terms_of_x_l1816_181607

theorem y_in_terms_of_x (x y : ℝ) (h : x + y = -1) : y = -1 - x := by
  sorry

end y_in_terms_of_x_l1816_181607


namespace balls_after_2010_steps_l1816_181613

/-- Converts a natural number to its base-6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Sums the digits in a list --/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

/-- Represents the ball-and-box process --/
def ballBoxProcess (steps : ℕ) : ℕ :=
  sumDigits (toBase6 steps)

/-- Theorem stating that the number of balls after 2010 steps
    is equal to the sum of digits in the base-6 representation of 2010 --/
theorem balls_after_2010_steps :
  ballBoxProcess 2010 = 11 := by sorry

end balls_after_2010_steps_l1816_181613


namespace total_savings_calculation_l1816_181609

def chlorine_price : ℝ := 10
def soap_price : ℝ := 16
def wipes_price : ℝ := 8

def chlorine_discount1 : ℝ := 0.20
def chlorine_discount2 : ℝ := 0.10
def chlorine_discount3 : ℝ := 0.05

def soap_discount1 : ℝ := 0.25
def soap_discount2 : ℝ := 0.05

def wipes_discount1 : ℝ := 0.30
def wipes_discount2 : ℝ := 0.15
def wipes_discount3 : ℝ := 0.20

def chlorine_quantity : ℕ := 4
def soap_quantity : ℕ := 6
def wipes_quantity : ℕ := 8

theorem total_savings_calculation :
  let chlorine_final_price := chlorine_price * (1 - chlorine_discount1) * (1 - chlorine_discount2) * (1 - chlorine_discount3)
  let soap_final_price := soap_price * (1 - soap_discount1) * (1 - soap_discount2)
  let wipes_final_price := wipes_price * (1 - wipes_discount1) * (1 - wipes_discount2) * (1 - wipes_discount3)
  let chlorine_savings := (chlorine_price - chlorine_final_price) * chlorine_quantity
  let soap_savings := (soap_price - soap_final_price) * soap_quantity
  let wipes_savings := (wipes_price - wipes_final_price) * wipes_quantity
  chlorine_savings + soap_savings + wipes_savings = 73.776 := by sorry

end total_savings_calculation_l1816_181609


namespace revenue_decrease_l1816_181644

theorem revenue_decrease (projected_increase : ℝ) (actual_vs_projected : ℝ) 
  (h1 : projected_increase = 0.20)
  (h2 : actual_vs_projected = 0.625) : 
  (1 - actual_vs_projected * (1 + projected_increase)) * 100 = 25 := by
  sorry

end revenue_decrease_l1816_181644


namespace perfect_square_condition_l1816_181608

theorem perfect_square_condition (a b : ℤ) :
  (∀ m n : ℕ, ∃ k : ℤ, a * m^2 + b * n^2 = k^2) →
  a * b = 0 := by
sorry

end perfect_square_condition_l1816_181608


namespace incorrect_transformation_l1816_181601

theorem incorrect_transformation (a b : ℝ) (h : a > b) : ¬(3 - a > 3 - b) := by
  sorry

end incorrect_transformation_l1816_181601


namespace comparison_theorems_l1816_181668

theorem comparison_theorems :
  (∀ a b : ℝ, a - b = 4 → a > b) ∧
  (∀ a b : ℝ, a - b = -2 → a < b) ∧
  (∀ x : ℝ, x > 0 → -x + 5 > -2*x + 4) ∧
  (∀ x y : ℝ, 
    (y > x → 5*x + 13*y + 2 > 6*x + 12*y + 2) ∧
    (y = x → 5*x + 13*y + 2 = 6*x + 12*y + 2) ∧
    (y < x → 5*x + 13*y + 2 < 6*x + 12*y + 2)) :=
by sorry

end comparison_theorems_l1816_181668


namespace nabla_two_three_l1816_181696

def nabla (a b : ℕ+) : ℕ := a.val ^ b.val * b.val ^ a.val

theorem nabla_two_three : nabla 2 3 = 72 := by sorry

end nabla_two_three_l1816_181696


namespace multiply_square_roots_l1816_181614

theorem multiply_square_roots : -2 * Real.sqrt 10 * (3 * Real.sqrt 30) = -60 * Real.sqrt 3 := by
  sorry

end multiply_square_roots_l1816_181614


namespace football_likers_l1816_181694

theorem football_likers (total : ℕ) (likers : ℕ) (players : ℕ) : 
  (24 : ℚ) / total = (likers : ℚ) / 250 →
  (players : ℚ) / likers = 1 / 2 →
  players = 50 →
  total = 60 := by
sorry

end football_likers_l1816_181694


namespace polynomial_coefficient_sum_l1816_181654

theorem polynomial_coefficient_sum (A B C D : ℚ) : 
  (∀ x : ℚ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 36 := by
sorry

end polynomial_coefficient_sum_l1816_181654


namespace count_rectangles_3x3_grid_l1816_181631

/-- The number of rectangles that can be formed on a 3x3 grid -/
def rectangles_on_3x3_grid : ℕ := 9

/-- Theorem stating that the number of rectangles on a 3x3 grid is 9 -/
theorem count_rectangles_3x3_grid : 
  rectangles_on_3x3_grid = 9 := by sorry

end count_rectangles_3x3_grid_l1816_181631


namespace harry_total_cost_l1816_181692

-- Define the conversion rate
def silver_per_gold : ℕ := 9

-- Define the costs
def spellbook_cost_gold : ℕ := 5
def potion_kit_cost_silver : ℕ := 20
def owl_cost_gold : ℕ := 28

-- Define the quantities
def num_spellbooks : ℕ := 5
def num_potion_kits : ℕ := 3
def num_owls : ℕ := 1

-- Define the total cost function
def total_cost_silver : ℕ :=
  (num_spellbooks * spellbook_cost_gold * silver_per_gold) +
  (num_potion_kits * potion_kit_cost_silver) +
  (num_owls * owl_cost_gold * silver_per_gold)

-- Theorem statement
theorem harry_total_cost :
  total_cost_silver = 537 :=
by sorry

end harry_total_cost_l1816_181692


namespace largest_integer_with_remainder_l1816_181630

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n := by
  sorry

end largest_integer_with_remainder_l1816_181630


namespace hyperbola_a_value_l1816_181652

/-- The focal length of a hyperbola -/
def focal_length (c : ℝ) : ℝ := 2 * c

/-- The equation of a hyperbola -/
def is_hyperbola (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

theorem hyperbola_a_value (a : ℝ) :
  a > 0 →
  focal_length (Real.sqrt 10) = 2 * Real.sqrt 10 →
  is_hyperbola a (Real.sqrt 6) (Real.sqrt 10) →
  a = 2 := by
  sorry

end hyperbola_a_value_l1816_181652


namespace tangent_line_equation_l1816_181693

/-- A line parallel to 2x - y + 1 = 0 and tangent to x^2 + y^2 = 5 has equation 2x - y ± 5 = 0 -/
theorem tangent_line_equation (x y : ℝ) :
  ∃ (k : ℝ), k = 5 ∨ k = -5 ∧
  (∀ (x y : ℝ), 2*x - y + k = 0 →
    (∀ (x₀ y₀ : ℝ), 2*x₀ - y₀ + 1 = 0 → ∃ (t : ℝ), x = x₀ + 2*t ∧ y = y₀ + t) ∧
    (∃! (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 5 ∧ 2*x₀ - y₀ + k = 0)) :=
by sorry

end tangent_line_equation_l1816_181693


namespace euler_triangle_inequality_l1816_181676

/-- 
For any triangle, let:
  r : radius of the incircle
  R : radius of the circumcircle
  d : distance between the incenter and circumcenter

Then, R ≥ 2r
-/
theorem euler_triangle_inequality (r R d : ℝ) : r > 0 → R > 0 → d > 0 → R ≥ 2 * r := by
  sorry

end euler_triangle_inequality_l1816_181676


namespace initial_candies_count_l1816_181675

/-- The number of candies sold on Day 1 -/
def day1_sales : ℕ := 1249

/-- The additional number of candies sold on Day 2 compared to Day 1 -/
def day2_additional : ℕ := 328

/-- The additional number of candies sold on Day 3 compared to Day 2 -/
def day3_additional : ℕ := 275

/-- The number of candies remaining after three days of sales -/
def remaining_candies : ℕ := 367

/-- The total number of candies at the beginning -/
def initial_candies : ℕ := day1_sales + (day1_sales + day2_additional) + (day1_sales + day2_additional + day3_additional) + remaining_candies

theorem initial_candies_count : initial_candies = 5045 := by
  sorry

end initial_candies_count_l1816_181675


namespace function_satisfying_divisibility_l1816_181638

theorem function_satisfying_divisibility (f : ℕ+ → ℕ+) :
  (∀ a b : ℕ+, (f a + b) ∣ (a^2 + f a * f b)) →
  ∀ n : ℕ+, f n = n :=
by sorry

end function_satisfying_divisibility_l1816_181638


namespace ginos_white_bears_l1816_181636

theorem ginos_white_bears :
  ∀ (brown_bears white_bears black_bears total_bears : ℕ),
    brown_bears = 15 →
    black_bears = 27 →
    total_bears = 66 →
    total_bears = brown_bears + white_bears + black_bears →
    white_bears = 24 := by
  sorry

end ginos_white_bears_l1816_181636


namespace power_of_two_equation_l1816_181642

theorem power_of_two_equation (N : ℕ) : (32^5 * 16^4) / 8^7 = 2^N → N = 20 := by
  sorry

end power_of_two_equation_l1816_181642


namespace square_difference_equality_l1816_181619

theorem square_difference_equality : 1013^2 - 1009^2 - 1011^2 + 997^2 = -19924 := by
  sorry

end square_difference_equality_l1816_181619


namespace paths_via_checkpoint_count_l1816_181610

/-- Number of paths in a grid from (0,0) to (a,b) -/
def gridPaths (a b : ℕ) : ℕ := Nat.choose (a + b) a

/-- The coordinates of point A -/
def A : ℕ × ℕ := (0, 0)

/-- The coordinates of point B -/
def B : ℕ × ℕ := (5, 4)

/-- The coordinates of checkpoint C -/
def C : ℕ × ℕ := (3, 2)

/-- The number of paths from A to B via C -/
def pathsViaCPoint : ℕ := 
  (gridPaths (C.1 - A.1) (C.2 - A.2)) * (gridPaths (B.1 - C.1) (B.2 - C.2))

theorem paths_via_checkpoint_count : pathsViaCPoint = 60 := by sorry

end paths_via_checkpoint_count_l1816_181610


namespace peach_pie_slices_count_l1816_181653

/-- Represents the number of slices in a peach pie -/
def peach_pie_slices : ℕ := sorry

/-- Represents the number of slices in an apple pie -/
def apple_pie_slices : ℕ := 8

/-- Represents the number of customers who ordered apple pie slices -/
def apple_pie_customers : ℕ := 56

/-- Represents the number of customers who ordered peach pie slices -/
def peach_pie_customers : ℕ := 48

/-- Represents the total number of pies sold during the weekend -/
def total_pies_sold : ℕ := 15

theorem peach_pie_slices_count : peach_pie_slices = 6 := by
  sorry

end peach_pie_slices_count_l1816_181653


namespace marble_ratio_theorem_l1816_181650

/-- Represents the number of marbles Elsa has at different points in the day -/
structure MarbleCount where
  initial : ℕ
  after_breakfast : ℕ
  after_lunch : ℕ
  after_mom_purchase : ℕ
  final : ℕ

/-- Represents the marble transactions throughout the day -/
structure MarbleTransactions where
  lost_at_breakfast : ℕ
  given_to_susie : ℕ
  bought_by_mom : ℕ

/-- Theorem stating the ratio of marbles Susie gave back to Elsa to the marbles Elsa gave to Susie -/
theorem marble_ratio_theorem (m : MarbleCount) (t : MarbleTransactions) : 
  m.initial = 40 →
  t.lost_at_breakfast = 3 →
  t.given_to_susie = 5 →
  t.bought_by_mom = 12 →
  m.final = 54 →
  m.after_breakfast = m.initial - t.lost_at_breakfast →
  m.after_lunch = m.after_breakfast - t.given_to_susie →
  m.after_mom_purchase = m.after_lunch + t.bought_by_mom →
  (m.final - m.after_mom_purchase) / t.given_to_susie = 2 :=
by sorry

end marble_ratio_theorem_l1816_181650


namespace range_of_a_for_zero_point_solution_for_specific_a_l1816_181665

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

-- Theorem for the range of a
theorem range_of_a_for_zero_point :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ (Set.Ioo (-1) 1) ∧ f a x = 0) ↔
    a ∈ Set.Icc (12 * (27 - 4 * Real.sqrt 6) / 211) (12 * (27 + 4 * Real.sqrt 6) / 211) :=
sorry

-- Theorem for the specific solution when a = 32/17
theorem solution_for_specific_a :
  ∃ x : ℝ, x ∈ (Set.Ioo (-1) 1) ∧ f (32/17) x = 0 ∧ x = 1/2 :=
sorry

end range_of_a_for_zero_point_solution_for_specific_a_l1816_181665


namespace pentagonal_prism_with_pyramid_sum_l1816_181645

/-- A shape formed by adding a pyramid to one pentagonal face of a pentagonal prism -/
structure PentagonalPrismWithPyramid where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- The sum of faces, vertices, and edges for a PentagonalPrismWithPyramid -/
def PentagonalPrismWithPyramid.sum (shape : PentagonalPrismWithPyramid) : ℕ :=
  shape.faces + shape.vertices + shape.edges

theorem pentagonal_prism_with_pyramid_sum :
  ∃ (shape : PentagonalPrismWithPyramid), shape.sum = 42 :=
sorry

end pentagonal_prism_with_pyramid_sum_l1816_181645


namespace intersection_x_coordinate_l1816_181672

theorem intersection_x_coordinate (x y : ℤ) : 
  (y ≡ 3 * x + 4 [ZMOD 9]) → 
  (y ≡ 7 * x + 2 [ZMOD 9]) → 
  (x ≡ 5 [ZMOD 9]) := by
sorry

end intersection_x_coordinate_l1816_181672


namespace min_value_theorem_l1816_181616

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : 2*a + b = 2) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = 2 → 1/a + 2/b ≤ 1/x + 2/y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = 2 ∧ 1/x + 2/y = 4) :=
sorry

end min_value_theorem_l1816_181616


namespace womens_average_age_l1816_181632

/-- The average age of two women given the following conditions:
    - There are initially 6 men
    - Two men aged 10 and 12 are replaced by two women
    - The average age increases by 2 years after the replacement
-/
theorem womens_average_age (initial_men : ℕ) (age_increase : ℝ) 
  (replaced_man1_age replaced_man2_age : ℕ) :
  initial_men = 6 →
  age_increase = 2 →
  replaced_man1_age = 10 →
  replaced_man2_age = 12 →
  ∃ (initial_avg : ℝ),
    ((initial_men : ℝ) * initial_avg - (replaced_man1_age + replaced_man2_age : ℝ) + 
     2 * ((initial_avg + age_increase) : ℝ)) / 2 = 17 :=
by sorry

end womens_average_age_l1816_181632


namespace eggs_in_boxes_l1816_181664

theorem eggs_in_boxes (eggs_per_box : ℕ) (num_boxes : ℕ) :
  eggs_per_box = 15 → num_boxes = 7 → eggs_per_box * num_boxes = 105 := by
  sorry

end eggs_in_boxes_l1816_181664


namespace quadratic_roots_sum_product_l1816_181603

theorem quadratic_roots_sum_product (p q : ℝ) (k : ℕ+) :
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0) →
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x + y = 2) →
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x * y = k) →
  p = -2 :=
by sorry

end quadratic_roots_sum_product_l1816_181603


namespace no_integer_roots_l1816_181640

theorem no_integer_roots (a b : ℤ) : ¬ ∃ x : ℤ, x^2 + 3*a*x + 3*(2 - b^2) = 0 := by
  sorry

end no_integer_roots_l1816_181640


namespace fruit_shop_problem_l1816_181625

theorem fruit_shop_problem (total_cost : ℕ) (total_profit : ℕ) 
  (lychee_cost : ℕ) (longan_cost : ℕ) (lychee_price : ℕ) (longan_price : ℕ) 
  (second_profit : ℕ) :
  total_cost = 3900 →
  total_profit = 1200 →
  lychee_cost = 120 →
  longan_cost = 100 →
  lychee_price = 150 →
  longan_price = 140 →
  second_profit = 960 →
  ∃ (lychee_boxes longan_boxes : ℕ) (discount_rate : ℚ),
    lychee_cost * lychee_boxes + longan_cost * longan_boxes = total_cost ∧
    (lychee_price - lychee_cost) * lychee_boxes + (longan_price - longan_cost) * longan_boxes = total_profit ∧
    lychee_boxes = 20 ∧
    longan_boxes = 15 ∧
    (lychee_price - lychee_cost) * lychee_boxes + 
      (longan_price * discount_rate - longan_cost) * (2 * longan_boxes) = second_profit ∧
    discount_rate = 4/5 := by
  sorry

end fruit_shop_problem_l1816_181625


namespace mary_payment_l1816_181635

def apple_cost : ℕ := 1
def orange_cost : ℕ := 2
def banana_cost : ℕ := 3
def discount_threshold : ℕ := 5
def discount_amount : ℕ := 1

def mary_apples : ℕ := 5
def mary_oranges : ℕ := 3
def mary_bananas : ℕ := 2

def total_fruits : ℕ := mary_apples + mary_oranges + mary_bananas

def fruit_cost : ℕ := mary_apples * apple_cost + mary_oranges * orange_cost + mary_bananas * banana_cost

def discount_sets : ℕ := total_fruits / discount_threshold

def total_discount : ℕ := discount_sets * discount_amount

def final_cost : ℕ := fruit_cost - total_discount

theorem mary_payment : final_cost = 15 := by
  sorry

end mary_payment_l1816_181635


namespace prob_sum_three_eq_one_over_216_l1816_181695

/-- The probability of rolling a specific number on a standard die -/
def prob_single_die : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The target sum we're looking for -/
def target_sum : ℕ := 3

/-- The probability of rolling a sum of 3 with three standard dice -/
def prob_sum_three : ℚ := (prob_single_die) ^ num_dice

theorem prob_sum_three_eq_one_over_216 : 
  prob_sum_three = 1 / 216 := by sorry

end prob_sum_three_eq_one_over_216_l1816_181695


namespace jessica_remaining_seashells_l1816_181673

/-- The number of seashells Jessica initially found -/
def initial_seashells : ℕ := 8

/-- The number of seashells Jessica gave to Joan -/
def given_seashells : ℕ := 6

/-- The number of seashells Jessica is left with -/
def remaining_seashells : ℕ := initial_seashells - given_seashells

theorem jessica_remaining_seashells : remaining_seashells = 2 := by
  sorry

end jessica_remaining_seashells_l1816_181673


namespace nested_average_calculation_l1816_181646

def average (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem nested_average_calculation : 
  let x := average 2 3 1
  let y := average 4 1 0
  average x y 5 = 26 / 9 := by sorry

end nested_average_calculation_l1816_181646


namespace ball_probabilities_l1816_181623

/-- Represents a bag of balls with a given number of black and white balls. -/
structure BagOfBalls where
  blackBalls : ℕ
  whiteBalls : ℕ

/-- Calculates the probability of drawing two black balls without replacement. -/
def probabilityTwoBlackBalls (bag : BagOfBalls) : ℚ :=
  let totalBalls := bag.blackBalls + bag.whiteBalls
  (bag.blackBalls.choose 2 : ℚ) / (totalBalls.choose 2)

/-- Calculates the probability of drawing a black ball on the second draw,
    given that a black ball was drawn on the first draw. -/
def probabilitySecondBlackGivenFirstBlack (bag : BagOfBalls) : ℚ :=
  (bag.blackBalls - 1 : ℚ) / (bag.blackBalls + bag.whiteBalls - 1)

theorem ball_probabilities (bag : BagOfBalls) 
  (h1 : bag.blackBalls = 6) (h2 : bag.whiteBalls = 4) : 
  probabilityTwoBlackBalls bag = 1/3 ∧ 
  probabilitySecondBlackGivenFirstBlack bag = 5/9 := by
  sorry


end ball_probabilities_l1816_181623


namespace f_properties_l1816_181604

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties :
  let max_value : ℝ := 1 / Real.exp 1
  ∀ (x₁ x₂ x₀ m : ℝ),
  (∀ x > 0, f x = (Real.log x) / x) →
  (∀ x > 0, f x ≤ max_value) →
  (f (Real.exp 1) = max_value) →
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f (Real.exp 1 + x) > f (Real.exp 1 - x)) →
  (f x₁ = m) →
  (f x₂ = m) →
  (x₀ = (x₁ + x₂) / 2) →
  (deriv f x₀ < 0) :=
by sorry

end f_properties_l1816_181604


namespace average_price_per_book_l1816_181685

theorem average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℕ) 
  (h1 : books1 = 32)
  (h2 : books2 = 60)
  (h3 : price1 = 1500)
  (h4 : price2 = 340) :
  (price1 + price2) / (books1 + books2) = 20 := by
sorry

end average_price_per_book_l1816_181685


namespace eggs_left_proof_l1816_181621

def eggs_left (initial : ℕ) (harry_takes : ℕ) (jenny_takes : ℕ) : ℕ :=
  initial - (harry_takes + jenny_takes)

theorem eggs_left_proof :
  eggs_left 47 5 8 = 34 := by
  sorry

end eggs_left_proof_l1816_181621


namespace opposite_sign_coordinates_second_quadrant_range_l1816_181687

def P (x : ℝ) : ℝ × ℝ := (x - 2, x)

theorem opposite_sign_coordinates (x : ℝ) :
  (P x).1 * (P x).2 < 0 → x = 1 := by sorry

theorem second_quadrant_range (x : ℝ) :
  (P x).1 < 0 ∧ (P x).2 > 0 → 0 < x ∧ x < 2 := by sorry

end opposite_sign_coordinates_second_quadrant_range_l1816_181687


namespace interest_rate_proof_l1816_181679

/-- Represents the annual interest rate as a real number between 0 and 1 -/
def annual_interest_rate : ℝ := 0.05

/-- The initial principal amount in rupees -/
def principal : ℝ := 4800

/-- The final amount after 2 years in rupees -/
def final_amount : ℝ := 5292

/-- The number of years the money is invested -/
def time : ℕ := 2

/-- The number of times interest is compounded per year -/
def compounds_per_year : ℕ := 1

theorem interest_rate_proof :
  final_amount = principal * (1 + annual_interest_rate) ^ (compounds_per_year * time) :=
sorry

end interest_rate_proof_l1816_181679


namespace new_england_population_l1816_181666

/-- The population of New York -/
def population_NY : ℕ := sorry

/-- The population of New England -/
def population_NE : ℕ := sorry

/-- New York's population is two-thirds of New England's -/
axiom ny_two_thirds_ne : population_NY = (2 * population_NE) / 3

/-- The combined population of New York and New England is 3,500,000 -/
axiom combined_population : population_NY + population_NE = 3500000

/-- Theorem: The population of New England is 2,100,000 -/
theorem new_england_population : population_NE = 2100000 := by sorry

end new_england_population_l1816_181666


namespace count_ten_digit_numbers_theorem_l1816_181655

/-- Count of ten-digit numbers with a given digit sum -/
def count_ten_digit_numbers (n : ℕ) : ℕ :=
  match n with
  | 2 => 46
  | 3 => 166
  | 4 => 361
  | _ => 0

/-- Theorem stating the count of ten-digit numbers with specific digit sums -/
theorem count_ten_digit_numbers_theorem :
  (count_ten_digit_numbers 2 = 46) ∧
  (count_ten_digit_numbers 3 = 166) ∧
  (count_ten_digit_numbers 4 = 361) := by
  sorry

end count_ten_digit_numbers_theorem_l1816_181655


namespace tangent_dot_product_l1816_181689

/-- The circle with center at the origin and radius 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Point M outside the circle -/
def M : ℝ × ℝ := (2, 0)

/-- A point is on the circle -/
def on_circle (p : ℝ × ℝ) : Prop := unit_circle p.1 p.2

/-- A line is tangent to the circle at a point -/
def is_tangent (p q : ℝ × ℝ) : Prop :=
  on_circle p ∧ (p.1 * q.1 + p.2 * q.2 = 1)

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem tangent_dot_product :
  ∃ (A B : ℝ × ℝ),
    on_circle A ∧
    on_circle B ∧
    is_tangent A M ∧
    is_tangent B M ∧
    dot_product (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2) = 3/2 := by
  sorry

end tangent_dot_product_l1816_181689


namespace train_speed_l1816_181634

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  bridge_length = 205 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry


end train_speed_l1816_181634


namespace non_zero_digits_count_l1816_181671

/-- The fraction we're working with -/
def f : ℚ := 120 / (2^5 * 5^9)

/-- Count of non-zero digits after the decimal point in the decimal representation of a rational number -/
noncomputable def count_non_zero_digits_after_decimal (q : ℚ) : ℕ := sorry

/-- The main theorem: the count of non-zero digits after the decimal point for our fraction is 2 -/
theorem non_zero_digits_count : count_non_zero_digits_after_decimal f = 2 := by sorry

end non_zero_digits_count_l1816_181671


namespace water_transfer_equilibrium_l1816_181624

theorem water_transfer_equilibrium (total : ℕ) (a b : ℕ) : 
  total = 48 →
  a = 30 →
  b = 18 →
  a + b = total →
  let a' := a - 2 * a
  let b' := b + 2 * a
  let a'' := a' + 2 * a'
  let b'' := b' - 2 * a'
  a'' = b'' := by sorry

end water_transfer_equilibrium_l1816_181624


namespace equation_solution_and_expression_value_l1816_181683

theorem equation_solution_and_expression_value :
  ∃ y : ℝ, (4 * y - 8 = 2 * y + 18) ∧ (3 * (y^2 + 6 * y + 12) = 777) := by
  sorry

end equation_solution_and_expression_value_l1816_181683


namespace parallel_linear_functions_theorem_l1816_181628

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b ∧ g x = a * x + c
  not_axis_parallel : ∀ (a b c : ℝ), (∀ x, f x = a * x + b ∧ g x = a * x + c) → a ≠ 0

/-- The condition that (f(x))^2 touches -6g(x) -/
def touches_neg_6g (p : ParallelLinearFunctions) : Prop :=
  ∃! x, (p.f x)^2 = -6 * p.g x

/-- The condition that (g(x))^2 touches Af(x) -/
def touches_Af (p : ParallelLinearFunctions) (A : ℝ) : Prop :=
  ∃! x, (p.g x)^2 = A * p.f x

/-- The main theorem -/
theorem parallel_linear_functions_theorem (p : ParallelLinearFunctions) 
  (h : touches_neg_6g p) : 
  ∀ A, touches_Af p A ↔ (A = 6 ∨ A = 0) := by
  sorry

end parallel_linear_functions_theorem_l1816_181628


namespace largest_mersenne_prime_under_500_l1816_181615

def is_mersenne_prime (m : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ m = 2^n - 1 ∧ Prime m

theorem largest_mersenne_prime_under_500 :
  (∀ m : ℕ, is_mersenne_prime m ∧ m < 500 → m ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end largest_mersenne_prime_under_500_l1816_181615
