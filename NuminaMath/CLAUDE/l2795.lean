import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_height_l2795_279576

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) : 
  area = base * height → area = 960 → base = 60 → height = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2795_279576


namespace NUMINAMATH_CALUDE_line_y_intercept_l2795_279502

/-- A line with slope 3 and x-intercept (-4, 0) has y-intercept (0, 12) -/
theorem line_y_intercept (slope : ℝ) (x_intercept : ℝ × ℝ) :
  slope = 3 ∧ x_intercept = (-4, 0) →
  ∃ (y : ℝ), (∀ (x : ℝ), y = slope * x + (slope * x_intercept.1 + x_intercept.2)) ∧
              y = slope * 0 + (slope * x_intercept.1 + x_intercept.2) ∧
              (0, y) = (0, 12) :=
by sorry


end NUMINAMATH_CALUDE_line_y_intercept_l2795_279502


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2795_279548

theorem quadratic_inequality_solution (a : ℝ) (h : a > 0) :
  let f := fun x => a * x^2 - (a^2 + 1) * x + a
  (∀ x, f x > 0 ↔
    (a > 1 ∧ (x < 1/a ∨ x > a)) ∨
    (a = 1 ∧ x ≠ 1) ∨
    (0 < a ∧ a < 1 ∧ (x < a ∨ x > 1/a))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2795_279548


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l2795_279594

theorem regular_polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) (h2 : 360 / n = 18) :
  (n - 2) * 180 = 3240 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l2795_279594


namespace NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l2795_279523

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting 
  (jack_speed Christina_speed lindy_speed : ℝ)
  (lindy_distance : ℝ)
  (h1 : jack_speed = 7)
  (h2 : Christina_speed = 8)
  (h3 : lindy_speed = 10)
  (h4 : lindy_distance = 100) :
  (jack_speed + Christina_speed) * (lindy_distance / lindy_speed) = 150 := by
  sorry

end NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l2795_279523


namespace NUMINAMATH_CALUDE_art_class_selection_l2795_279593

theorem art_class_selection (n m k : ℕ) (hn : n = 10) (hm : m = 4) (hk : k = 2) :
  (Nat.choose (n - k + 1) (m - k + 1)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_art_class_selection_l2795_279593


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2795_279552

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > |a n|

theorem condition_sufficient_not_necessary :
  (∀ a : ℕ → ℝ, satisfies_condition a → is_increasing a) ∧
  (∃ a : ℕ → ℝ, is_increasing a ∧ ¬satisfies_condition a) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2795_279552


namespace NUMINAMATH_CALUDE_cookout_buns_per_pack_alex_cookout_buns_per_pack_l2795_279521

/-- Calculates the number of buns in each pack given the cookout conditions -/
theorem cookout_buns_per_pack (total_guests : ℕ) (burgers_per_guest : ℕ) 
  (non_meat_guests : ℕ) (non_bread_guests : ℕ) (bun_packs : ℕ) : ℕ :=
  let guests_eating_meat := total_guests - non_meat_guests
  let guests_eating_bread := guests_eating_meat - non_bread_guests
  let total_buns_needed := guests_eating_bread * burgers_per_guest
  total_buns_needed / bun_packs

/-- Proves that the number of buns in each pack for Alex's cookout is 8 -/
theorem alex_cookout_buns_per_pack : 
  cookout_buns_per_pack 10 3 1 1 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cookout_buns_per_pack_alex_cookout_buns_per_pack_l2795_279521


namespace NUMINAMATH_CALUDE_total_followers_count_l2795_279528

def instagram_followers : ℕ := 240
def facebook_followers : ℕ := 500

def twitter_followers : ℕ := (instagram_followers + facebook_followers) / 2
def tiktok_followers : ℕ := 3 * twitter_followers
def youtube_followers : ℕ := tiktok_followers + 510

def total_followers : ℕ := instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers

theorem total_followers_count : total_followers = 3840 := by
  sorry

end NUMINAMATH_CALUDE_total_followers_count_l2795_279528


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2795_279577

theorem inequality_solution_set : 
  {x : ℝ | x^2 - 7*x + 12 < 0} = Set.Ioo 3 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2795_279577


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2795_279508

theorem diophantine_equation_solution :
  ∀ x y z : ℤ, 2*x^2 + 2*x^2*z^2 + z^2 + 7*y^2 - 42*y + 33 = 0 ↔
  (x = 1 ∧ y = 5 ∧ z = 0) ∨
  (x = -1 ∧ y = 5 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 0) ∨
  (x = -1 ∧ y = 1 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2795_279508


namespace NUMINAMATH_CALUDE_angle_sum_equality_l2795_279516

theorem angle_sum_equality (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h4 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equality_l2795_279516


namespace NUMINAMATH_CALUDE_parabola_coefficients_l2795_279542

/-- A parabola with given properties has specific coefficients -/
theorem parabola_coefficients :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (3 = a * 4^2 + b * 4 + c) →
  (4 = -b / (2 * a)) →
  (7 = a * 2^2 + b * 2 + c) →
  a = 1 ∧ b = -8 ∧ c = 19 :=
by sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l2795_279542


namespace NUMINAMATH_CALUDE_circle_tangent_slope_range_l2795_279510

theorem circle_tangent_slope_range (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (k : ℝ), k = 3/4 ∧ ∀ (z : ℝ), z ≥ k → ∃ (a b : ℝ), a^2 + b^2 = 1 ∧ z = (b + 2) / (a + 1) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_slope_range_l2795_279510


namespace NUMINAMATH_CALUDE_jerry_needs_72_dollars_l2795_279531

/-- The amount of money Jerry needs to finish his action figure collection -/
def jerryNeedsMoney (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Proof that Jerry needs $72 to finish his collection -/
theorem jerry_needs_72_dollars :
  jerryNeedsMoney 7 16 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_jerry_needs_72_dollars_l2795_279531


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l2795_279573

theorem hot_dogs_remainder : 34582918 % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l2795_279573


namespace NUMINAMATH_CALUDE_grid_bottom_right_value_l2795_279572

/-- Represents a 4x4 grid of rational numbers -/
def Grid := Fin 4 → Fin 4 → ℚ

/-- Checks if a sequence of 4 rational numbers forms an arithmetic progression -/
def isArithmeticSequence (s : Fin 4 → ℚ) : Prop :=
  ∃ d : ℚ, ∀ i : Fin 3, s (i + 1) - s i = d

/-- A grid satisfying the problem conditions -/
def validGrid (g : Grid) : Prop :=
  (∀ i : Fin 4, isArithmeticSequence (λ j ↦ g i j)) ∧  -- Each row is an arithmetic sequence
  (∀ j : Fin 4, isArithmeticSequence (λ i ↦ g i j)) ∧  -- Each column is an arithmetic sequence
  g 0 0 = 1 ∧ g 1 0 = 4 ∧ g 2 0 = 7 ∧ g 3 0 = 10 ∧     -- First column values
  g 2 3 = 25 ∧ g 3 2 = 36                              -- Given values in the grid

theorem grid_bottom_right_value (g : Grid) (h : validGrid g) : g 3 3 = 37 := by
  sorry

end NUMINAMATH_CALUDE_grid_bottom_right_value_l2795_279572


namespace NUMINAMATH_CALUDE_roots_equation_value_l2795_279549

theorem roots_equation_value (α β : ℝ) : 
  (α^2 + 2*α - 1 = 0) → 
  (β^2 + 2*β - 1 = 0) → 
  α^2 + 3*α + β = -1 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_value_l2795_279549


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2795_279563

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | x * (x - 1) < 0}

-- Theorem statement
theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2795_279563


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2795_279578

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + a ≥ 0) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2795_279578


namespace NUMINAMATH_CALUDE_union_of_sets_l2795_279554

theorem union_of_sets : 
  let A : Set ℕ := {1, 3, 5}
  let B : Set ℕ := {3, 5, 7}
  A ∪ B = {1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l2795_279554


namespace NUMINAMATH_CALUDE_henry_seed_growth_l2795_279501

/-- Given that Henry starts with 5 seeds and triples his seeds each day, 
    this theorem proves that it takes 6 days to exceed 500 seeds. -/
theorem henry_seed_growth (n : ℕ) : n > 0 ∧ 5 * 3^(n-1) > 500 ↔ n ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_henry_seed_growth_l2795_279501


namespace NUMINAMATH_CALUDE_antonios_meatballs_l2795_279540

/-- Given the conditions of Antonio's meatball preparation, prove the amount of hamburger per meatball. -/
theorem antonios_meatballs (family_members : ℕ) (total_hamburger : ℝ) (antonios_meatballs : ℕ) :
  family_members = 8 →
  total_hamburger = 4 →
  antonios_meatballs = 4 →
  (total_hamburger / (family_members * antonios_meatballs) : ℝ) = 0.125 := by
  sorry

#check antonios_meatballs

end NUMINAMATH_CALUDE_antonios_meatballs_l2795_279540


namespace NUMINAMATH_CALUDE_cone_volume_with_plane_intersection_l2795_279517

/-- The volume of a cone given specific plane intersections -/
theorem cone_volume_with_plane_intersection 
  (p q : ℝ) (a α : ℝ) (hp : p > 0) (hq : q > 0) (ha : a > 0) (hα : 0 < α ∧ α < π / 2) :
  let V := (π * a^3) / (3 * Real.sin α * Real.cos α^2 * Real.cos (π * q / (p + q))^2)
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧ V = (1/3) * π * r^2 * h :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_with_plane_intersection_l2795_279517


namespace NUMINAMATH_CALUDE_division_in_ratio_l2795_279586

theorem division_in_ratio (total : ℕ) (x_ratio y_ratio : ℕ) (x_amount : ℕ) : 
  total = 5000 → 
  x_ratio = 2 → 
  y_ratio = 8 → 
  x_amount = total * x_ratio / (x_ratio + y_ratio) → 
  x_amount = 1000 := by
sorry

end NUMINAMATH_CALUDE_division_in_ratio_l2795_279586


namespace NUMINAMATH_CALUDE_exists_problem_solved_by_half_not_all_l2795_279560

/-- Represents a jury member -/
structure JuryMember where
  id : Nat
  solved_problems : Finset Nat

/-- Represents the contest setup -/
structure ContestSetup where
  jury_members : Finset JuryMember
  total_problems : Nat
  problems_per_member : Nat

/-- Main theorem: There exists a problem solved by at least half but not all jury members -/
theorem exists_problem_solved_by_half_not_all (setup : ContestSetup)
  (h1 : setup.jury_members.card = 40)
  (h2 : setup.total_problems = 30)
  (h3 : setup.problems_per_member = 26)
  (h4 : ∀ m1 m2 : JuryMember, m1 ∈ setup.jury_members → m2 ∈ setup.jury_members → m1 ≠ m2 → m1.solved_problems ≠ m2.solved_problems) :
  ∃ p : Nat, p < setup.total_problems ∧ 
    (20 ≤ (setup.jury_members.filter (λ m => p ∈ m.solved_problems)).card) ∧
    ((setup.jury_members.filter (λ m => p ∈ m.solved_problems)).card < 40) := by
  sorry


end NUMINAMATH_CALUDE_exists_problem_solved_by_half_not_all_l2795_279560


namespace NUMINAMATH_CALUDE_expected_digits_icosahedral_die_l2795_279507

def icosahedral_die := Finset.range 20

theorem expected_digits_icosahedral_die :
  let digits_function := fun n => if n < 10 then 1 else 2
  let expected_value := (icosahedral_die.sum fun i => digits_function (i + 1)) / icosahedral_die.card
  expected_value = 31 / 20 := by
sorry

end NUMINAMATH_CALUDE_expected_digits_icosahedral_die_l2795_279507


namespace NUMINAMATH_CALUDE_selling_price_calculation_l2795_279583

/-- The selling price that yields a 4% higher gain than selling at 340, given a cost of 250 -/
def higher_selling_price (cost : ℝ) (lower_price : ℝ) : ℝ :=
  let lower_gain := lower_price - cost
  let higher_gain := lower_gain * 1.04
  cost + higher_gain

theorem selling_price_calculation :
  higher_selling_price 250 340 = 343.6 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l2795_279583


namespace NUMINAMATH_CALUDE_tan_75_degrees_l2795_279597

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  -- We define tan 75° as tan(90° - 15°)
  have h1 : Real.tan (75 * π / 180) = Real.tan ((90 - 15) * π / 180) := by sorry
  
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_tan_75_degrees_l2795_279597


namespace NUMINAMATH_CALUDE_equation_relation_l2795_279596

theorem equation_relation (x y z w : ℝ) :
  (x + 2 * y) / (2 * y + 3 * z) = (3 * z + 4 * w) / (4 * w + x) →
  x = 3 * z ∨ x + 2 * y + 4 * w + 3 * z = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_relation_l2795_279596


namespace NUMINAMATH_CALUDE_polynomial_no_real_roots_l2795_279569

theorem polynomial_no_real_roots (a b c : ℝ) (h : |a| + |b| + |c| ≤ Real.sqrt 2) :
  ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 > 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_no_real_roots_l2795_279569


namespace NUMINAMATH_CALUDE_probability_sum_20_l2795_279539

/-- A die is represented as a finite set of natural numbers from 1 to 12 -/
def TwelveSidedDie : Finset ℕ := Finset.range 12 

/-- The sum of two dice rolls -/
def DiceSum (roll1 roll2 : ℕ) : ℕ := roll1 + roll2

/-- The set of all possible outcomes when rolling two dice -/
def AllOutcomes : Finset (ℕ × ℕ) := TwelveSidedDie.product TwelveSidedDie

/-- The set of favorable outcomes (sum of 20) -/
def FavorableOutcomes : Finset (ℕ × ℕ) :=
  AllOutcomes.filter (fun p => DiceSum p.1 p.2 = 20)

/-- The probability of rolling a sum of 20 with two twelve-sided dice -/
theorem probability_sum_20 : 
  (FavorableOutcomes.card : ℚ) / AllOutcomes.card = 5 / 144 := by
  sorry


end NUMINAMATH_CALUDE_probability_sum_20_l2795_279539


namespace NUMINAMATH_CALUDE_insurance_cost_calculation_l2795_279544

/-- Calculates the total annual insurance cost given quarterly, monthly, annual, and semi-annual payments -/
def total_annual_insurance_cost (car_quarterly : ℕ) (home_monthly : ℕ) (health_annual : ℕ) (life_semiannual : ℕ) : ℕ :=
  car_quarterly * 4 + home_monthly * 12 + health_annual + life_semiannual * 2

/-- Theorem stating that given specific insurance costs, the total annual cost is 8757 -/
theorem insurance_cost_calculation :
  total_annual_insurance_cost 378 125 5045 850 = 8757 := by
  sorry

end NUMINAMATH_CALUDE_insurance_cost_calculation_l2795_279544


namespace NUMINAMATH_CALUDE_no_x_squared_term_l2795_279551

theorem no_x_squared_term (m : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + 2 * x * y + y^2 + m * x^2 = 2 * x * y + y^2) ↔ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l2795_279551


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_20122012_l2795_279504

theorem no_integer_solutions_for_20122012 :
  ¬∃ (a b c : ℤ), a^2 + b^2 + c^2 = 20122012 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_20122012_l2795_279504


namespace NUMINAMATH_CALUDE_cookie_remainder_l2795_279580

theorem cookie_remainder (whole : ℝ) (person_a_fraction : ℝ) (person_b_fraction : ℝ) :
  person_a_fraction = 0.7 →
  person_b_fraction = 1/3 →
  (whole - person_a_fraction * whole) * (1 - person_b_fraction) = 0.2 * whole := by
  sorry

end NUMINAMATH_CALUDE_cookie_remainder_l2795_279580


namespace NUMINAMATH_CALUDE_line_through_points_l2795_279575

/-- A line passing through three given points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The y-coordinate of a point on the line given its x-coordinate -/
def y_coord (l : Line) (x : ℝ) : ℝ :=
  sorry

theorem line_through_points (l : Line) : 
  l.point1 = (2, 8) ∧ l.point2 = (4, 14) ∧ l.point3 = (6, 20) → 
  y_coord l 50 = 152 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2795_279575


namespace NUMINAMATH_CALUDE_whale_consumption_increase_l2795_279514

/-- Represents the whale's plankton consumption pattern over 9 hours -/
structure WhaleConsumption where
  initial : ℝ  -- Initial consumption in the first hour
  increase : ℝ  -- Increase in consumption each hour
  total : ℝ     -- Total consumption over 9 hours
  sixth_hour : ℝ -- Consumption in the sixth hour

/-- The whale's consumption satisfies the given conditions -/
def satisfies_conditions (w : WhaleConsumption) : Prop :=
  w.total = 270 ∧ 
  w.sixth_hour = 33 ∧ 
  w.total = (9 * w.initial + 36 * w.increase) ∧
  w.sixth_hour = w.initial + 5 * w.increase

/-- The theorem stating that the increase in consumption is 3 kilos per hour -/
theorem whale_consumption_increase (w : WhaleConsumption) 
  (h : satisfies_conditions w) : w.increase = 3 := by
  sorry

end NUMINAMATH_CALUDE_whale_consumption_increase_l2795_279514


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2795_279545

theorem inequality_solution_set (x : ℝ) :
  (∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y) ↔
  0 ≤ x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2795_279545


namespace NUMINAMATH_CALUDE_radio_price_rank_l2795_279562

theorem radio_price_rank (n : ℕ) (prices : Finset ℕ) (radio_price : ℕ) :
  n = 43 →
  prices.card = n →
  (∀ (p q : ℕ), p ∈ prices → q ∈ prices → p ≠ q) →
  radio_price ∈ prices →
  (prices.filter (λ p => p > radio_price)).card = 8 →
  ∃ (m : ℕ), (prices.filter (λ p => p < radio_price)).card = m - 1 →
  (prices.filter (λ p => p ≤ radio_price)).card = 35 :=
by sorry

end NUMINAMATH_CALUDE_radio_price_rank_l2795_279562


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l2795_279584

theorem largest_prime_factor_of_1001 : ∃ p : ℕ, p.Prime ∧ p ∣ 1001 ∧ ∀ q : ℕ, q.Prime → q ∣ 1001 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l2795_279584


namespace NUMINAMATH_CALUDE_largest_number_in_systematic_sample_l2795_279590

/-- The largest number in a systematic sample --/
theorem largest_number_in_systematic_sample :
  let population_size : ℕ := 60
  let sample_size : ℕ := 10
  let remainder : ℕ := 3
  let divisor : ℕ := 6
  let sampling_interval : ℕ := population_size / sample_size
  let first_sample : ℕ := remainder
  let last_sample : ℕ := first_sample + sampling_interval * (sample_size - 1)
  last_sample = 57 := by sorry

end NUMINAMATH_CALUDE_largest_number_in_systematic_sample_l2795_279590


namespace NUMINAMATH_CALUDE_initial_kittens_initial_kittens_is_18_l2795_279500

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica : ℕ := 3

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara : ℕ := 6

/-- The number of kittens Tim has left -/
def kittens_left : ℕ := 9

/-- Theorem stating the initial number of kittens Tim had -/
theorem initial_kittens : ℕ :=
  kittens_to_jessica + kittens_to_sara + kittens_left

/-- Proof that the initial number of kittens is 18 -/
theorem initial_kittens_is_18 : initial_kittens = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_kittens_initial_kittens_is_18_l2795_279500


namespace NUMINAMATH_CALUDE_share_distribution_l2795_279568

theorem share_distribution (total : ℕ) (a b c : ℚ) 
  (h1 : total = 880)
  (h2 : a + b + c = total)
  (h3 : 4 * a = 5 * b)
  (h4 : 5 * b = 10 * c) :
  c = 160 := by
  sorry

end NUMINAMATH_CALUDE_share_distribution_l2795_279568


namespace NUMINAMATH_CALUDE_four_people_seven_chairs_two_occupied_l2795_279567

/-- The number of ways to arrange n distinct objects in r positions --/
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways four people can sit in a row of seven chairs
    where two specific chairs are always occupied --/
theorem four_people_seven_chairs_two_occupied : 
  permutation 5 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_four_people_seven_chairs_two_occupied_l2795_279567


namespace NUMINAMATH_CALUDE_tan_theta_value_l2795_279566

theorem tan_theta_value (θ : Real) 
  (h1 : 2 * Real.sin θ + Real.cos θ = Real.sqrt 2 / 3)
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.tan θ = -(90 + 5 * Real.sqrt 86) / 168 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2795_279566


namespace NUMINAMATH_CALUDE_common_volume_formula_l2795_279532

/-- Represents a cube with edge length a -/
structure Cube where
  a : ℝ
  a_pos : a > 0

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- Represents the configuration of a cube and a tetrahedron with aligned edges and coinciding midpoints -/
structure CubeTetrahedronConfig where
  cube : Cube
  tetrahedron : RegularTetrahedron
  aligned_edges : Bool
  coinciding_midpoints : Bool

/-- Calculates the volume of the common part of a cube and a tetrahedron in the given configuration -/
def common_volume (config : CubeTetrahedronConfig) : ℝ := sorry

/-- Theorem stating the volume of the common part of the cube and tetrahedron -/
theorem common_volume_formula (config : CubeTetrahedronConfig) 
  (h_aligned : config.aligned_edges = true) 
  (h_coincide : config.coinciding_midpoints = true) :
  common_volume config = (config.cube.a^3 * Real.sqrt 2 / 12) * (16 * Real.sqrt 2 - 17) := by
  sorry

end NUMINAMATH_CALUDE_common_volume_formula_l2795_279532


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2795_279550

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x^2 < 2*x + 3 → x ≥ 0 ∧ 0^2 < 2*0 + 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2795_279550


namespace NUMINAMATH_CALUDE_root_sum_absolute_value_l2795_279505

theorem root_sum_absolute_value (m : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 106 := by
sorry

end NUMINAMATH_CALUDE_root_sum_absolute_value_l2795_279505


namespace NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l2795_279537

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y - 7 = 0
def line2 (x y : ℝ) : Prop := 7*x + 15*y + 1 = 0
def line3 (x y : ℝ) : Prop := x + 2*y - 3 = 0
def target_line (x y : ℝ) : Prop := 3*x + 6*y - 2 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Theorem statement
theorem line_through_intersection_and_parallel :
  ∃ (x y : ℝ), intersection_point x y ∧ target_line x y ∧
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), target_line x y ↔ line3 (k*x) (k*y) :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l2795_279537


namespace NUMINAMATH_CALUDE_circle_proof_l2795_279570

-- Define the points A and B
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (3, -1)

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 16

-- Theorem statement
theorem circle_proof :
  ∃ (center : ℝ × ℝ),
    center_line center.1 center.2 ∧
    circle_equation A.1 A.2 ∧
    circle_equation B.1 B.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_proof_l2795_279570


namespace NUMINAMATH_CALUDE_vector_properties_l2795_279588

def e₁ : ℝ × ℝ := (1, 0)
def e₂ : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (3 * e₁.1 - 2 * e₂.1, 3 * e₁.2 - 2 * e₂.2)
def b : ℝ × ℝ := (4 * e₁.1 + e₂.1, 4 * e₁.2 + e₂.2)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 10) ∧
  ((a.1 + b.1)^2 + (a.2 + b.2)^2 = 50) ∧
  ((a.1 * b.1 + a.2 * b.2)^2 = 100 * ((a.1^2 + a.2^2) * (b.1^2 + b.2^2)) / 221) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l2795_279588


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l2795_279564

theorem cosine_sine_identity : 
  Real.cos (80 * π / 180) * Real.cos (35 * π / 180) + 
  Real.sin (80 * π / 180) * Real.cos (55 * π / 180) = 
  (1 / 2) * (Real.sin (65 * π / 180) + Real.sin (25 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l2795_279564


namespace NUMINAMATH_CALUDE_point_product_l2795_279509

theorem point_product (y₁ y₂ : ℝ) : 
  ((-4 - 7)^2 + (y₁ - 3)^2 = 13^2) →
  ((-4 - 7)^2 + (y₂ - 3)^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -39 := by
sorry

end NUMINAMATH_CALUDE_point_product_l2795_279509


namespace NUMINAMATH_CALUDE_expected_min_swaps_value_l2795_279541

/-- Represents a pair of twins -/
structure TwinPair :=
  (twin1 : ℕ)
  (twin2 : ℕ)

/-- Represents an arrangement of twin pairs around a circle -/
def Arrangement := List TwinPair

/-- Computes whether an arrangement has adjacent twins -/
def has_adjacent_twins (arr : Arrangement) : Prop :=
  sorry

/-- Performs a swap between two adjacent positions in the arrangement -/
def swap (arr : Arrangement) (pos : ℕ) : Arrangement :=
  sorry

/-- Computes the minimum number of swaps needed to separate all twins -/
def min_swaps (arr : Arrangement) : ℕ :=
  sorry

/-- Generates all possible arrangements of 5 pairs of twins -/
def all_arrangements : List Arrangement :=
  sorry

/-- Computes the expected value of the minimum number of swaps -/
def expected_min_swaps : ℚ :=
  sorry

theorem expected_min_swaps_value : 
  expected_min_swaps = 926 / 945 :=
sorry

end NUMINAMATH_CALUDE_expected_min_swaps_value_l2795_279541


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l2795_279529

/-- The vertex of the parabola y = x^2 - 9 has coordinates (0, -9) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x => x^2 - 9
  ∃ (x y : ℝ), (∀ t, f t ≥ f x) ∧ y = f x ∧ x = 0 ∧ y = -9 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l2795_279529


namespace NUMINAMATH_CALUDE_dividend_calculation_l2795_279559

/-- Given a division problem with the following parameters:
    - x: Real number equal to 0.25
    - quotient: Function of x defined as (3/2)x - 2,175.4
    - divisor: Function of x defined as 20,147x² - 785
    - remainder: Function of x defined as (-1/4)x³ + 1,112.7
    
    This theorem states that the dividend, calculated as (divisor * quotient) + remainder,
    is approximately equal to -1,031,103.16 (rounded to two decimal places). -/
theorem dividend_calculation (x : ℝ) 
    (hx : x = 0.25)
    (quotient : ℝ → ℝ)
    (hquotient : quotient = fun y => (3/2)*y - 2175.4)
    (divisor : ℝ → ℝ)
    (hdivisor : divisor = fun y => 20147*y^2 - 785)
    (remainder : ℝ → ℝ)
    (hremainder : remainder = fun y => (-1/4)*y^3 + 1112.7)
    : ∃ ε > 0, |((divisor x) * (quotient x) + (remainder x)) - (-1031103.16)| < ε :=
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2795_279559


namespace NUMINAMATH_CALUDE_roots_square_sum_l2795_279520

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := x^2 - 3*x + 1

-- Define the roots
theorem roots_square_sum : 
  ∀ r s : ℝ, quadratic r = 0 → quadratic s = 0 → r^2 + s^2 = 7 :=
by
  sorry

#check roots_square_sum

end NUMINAMATH_CALUDE_roots_square_sum_l2795_279520


namespace NUMINAMATH_CALUDE_mary_sugar_already_added_l2795_279525

/-- Given a recipe that requires a total amount of sugar and the amount still needed to be added,
    calculate the amount of sugar already put in. -/
def sugar_already_added (total_required : ℕ) (still_needed : ℕ) : ℕ :=
  total_required - still_needed

/-- Theorem stating that given the specific values from the problem,
    the amount of sugar already added is 2 cups. -/
theorem mary_sugar_already_added :
  sugar_already_added 13 11 = 2 := by sorry

end NUMINAMATH_CALUDE_mary_sugar_already_added_l2795_279525


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l2795_279519

def expression (x : ℝ) : ℝ := 
  3 * (x^2 - x^3 + x) + 3 * (x + 2*x^3 - 3*x^2 + 3*x^5 + x^3) - 5 * (1 + x - 4*x^3 - x^2)

theorem coefficient_of_x_cubed : 
  ∃ (a b c d e : ℝ), expression x = a*x^5 + b*x^4 + 26*x^3 + c*x^2 + d*x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l2795_279519


namespace NUMINAMATH_CALUDE_max_ab_min_3x_4y_max_f_l2795_279589

-- Part 1
theorem max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  a * b ≤ 1 / 16 := by sorry

-- Part 2
theorem min_3x_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 := by sorry

-- Part 3
theorem max_f (x : ℝ) (h : x < 5 / 4) :
  4 * x - 2 + 1 / (4 * x - 5) ≤ 1 := by sorry

end NUMINAMATH_CALUDE_max_ab_min_3x_4y_max_f_l2795_279589


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l2795_279555

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 ∧ initial_mean = 180 ∧ incorrect_value = 135 ∧ correct_value = 155 →
  (n * initial_mean + (correct_value - incorrect_value)) / n = 180.67 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l2795_279555


namespace NUMINAMATH_CALUDE_card_area_theorem_l2795_279592

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem card_area_theorem (original : Rectangle) 
  (h1 : original.length = 3 ∧ original.width = 7)
  (h2 : ∃ (shortened : Rectangle), 
    (shortened.length = original.length ∧ shortened.width = original.width - 2) ∨
    (shortened.length = original.length - 2 ∧ shortened.width = original.width) ∧
    area shortened = 15) :
  ∃ (other_shortened : Rectangle),
    ((other_shortened.length = original.length - 1 ∧ other_shortened.width = original.width) ∨
     (other_shortened.length = original.length ∧ other_shortened.width = original.width - 1)) ∧
    area other_shortened = 10 := by
  sorry

end NUMINAMATH_CALUDE_card_area_theorem_l2795_279592


namespace NUMINAMATH_CALUDE_carrots_in_second_bed_l2795_279547

/-- Given Kelly's carrot harvest information, prove the number of carrots in the second bed --/
theorem carrots_in_second_bed 
  (total_pounds : ℕ)
  (carrots_per_pound : ℕ)
  (first_bed : ℕ)
  (third_bed : ℕ)
  (h1 : total_pounds = 39)
  (h2 : carrots_per_pound = 6)
  (h3 : first_bed = 55)
  (h4 : third_bed = 78) :
  total_pounds * carrots_per_pound - first_bed - third_bed = 101 := by
  sorry

#check carrots_in_second_bed

end NUMINAMATH_CALUDE_carrots_in_second_bed_l2795_279547


namespace NUMINAMATH_CALUDE_binomial_product_factorial_equals_l2795_279581

theorem binomial_product_factorial_equals : (
  Nat.choose 10 3 * Nat.choose 8 3 * (Nat.factorial 7 / Nat.factorial 4)
) = 235200 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_factorial_equals_l2795_279581


namespace NUMINAMATH_CALUDE_largest_expression_l2795_279585

def P : ℕ := 3 * 2024^2025
def Q : ℕ := 2024^2025
def R : ℕ := 2023 * 2024^2024
def S : ℕ := 3 * 2024^2024
def T : ℕ := 2024^2024
def U : ℕ := 2024^2023

theorem largest_expression :
  (P - Q ≥ Q - R) ∧
  (P - Q ≥ R - S) ∧
  (P - Q ≥ S - T) ∧
  (P - Q ≥ T - U) :=
by sorry

end NUMINAMATH_CALUDE_largest_expression_l2795_279585


namespace NUMINAMATH_CALUDE_data_set_average_l2795_279587

theorem data_set_average (a : ℝ) : 
  let data_set := [4, 2*a, 3-a, 5, 6]
  (data_set.sum / data_set.length = 4) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_data_set_average_l2795_279587


namespace NUMINAMATH_CALUDE_power_product_equals_four_digit_l2795_279591

/-- Given that 2^x × 9^y equals the four-digit number 2x9y, prove that x^2 * y^3 = 200 -/
theorem power_product_equals_four_digit (x y : ℕ) : 
  (2^x * 9^y = 2000 + 100*x + 10*y + 9) → 
  (1000 ≤ 2000 + 100*x + 10*y + 9) → 
  (2000 + 100*x + 10*y + 9 < 10000) → 
  x^2 * y^3 = 200 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_four_digit_l2795_279591


namespace NUMINAMATH_CALUDE_singer_work_hours_l2795_279558

/-- Calculates the total hours taken to complete multiple songs given the daily work hours, days per song, and number of songs. -/
def totalHours (hoursPerDay : ℕ) (daysPerSong : ℕ) (numberOfSongs : ℕ) : ℕ :=
  hoursPerDay * daysPerSong * numberOfSongs

/-- Proves that a singer working 10 hours a day for 10 days on each of 3 songs will take 300 hours in total. -/
theorem singer_work_hours : totalHours 10 10 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_singer_work_hours_l2795_279558


namespace NUMINAMATH_CALUDE_set_intersection_complement_l2795_279506

/-- Given sets A and B, if the intersection of the complement of A and B equals B,
    then m is less than or equal to -11 or greater than or equal to 3. -/
theorem set_intersection_complement (m : ℝ) : 
  let A : Set ℝ := {x | -2 < x ∧ x < 3}
  let B : Set ℝ := {x | m < x ∧ x < m + 9}
  (Aᶜ ∩ B = B) → (m ≤ -11 ∨ m ≥ 3) :=
by
  sorry


end NUMINAMATH_CALUDE_set_intersection_complement_l2795_279506


namespace NUMINAMATH_CALUDE_solution_value_l2795_279536

theorem solution_value (a : ℝ) : (2 * a = 4) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2795_279536


namespace NUMINAMATH_CALUDE_remainder_theorem_l2795_279538

theorem remainder_theorem (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (hu : u = x / y) (hv : v = x % y) (hv_bound : v < y) : 
  (x + 3 * u * y + y) % y = v := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2795_279538


namespace NUMINAMATH_CALUDE_solution_set_f_max_value_g_range_of_m_l2795_279524

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the function g
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f (x : ℝ) : f x ≥ 1 ↔ x ≥ 1 := by sorry

-- Theorem for the maximum value of g(x)
theorem max_value_g : ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : (∃ (x : ℝ), f x ≥ x^2 - x + m) ↔ m ≤ 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_max_value_g_range_of_m_l2795_279524


namespace NUMINAMATH_CALUDE_probability_green_ball_l2795_279579

/-- The probability of drawing a green ball from a bag with specified contents -/
theorem probability_green_ball (green black red : ℕ) : 
  green = 3 → black = 3 → red = 6 → 
  (green : ℚ) / (green + black + red : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_ball_l2795_279579


namespace NUMINAMATH_CALUDE_photocopy_pages_theorem_l2795_279535

/-- The number of team members -/
def team_members : ℕ := 23

/-- The cost per page for the first 300 pages (in tenths of yuan) -/
def cost_first_300 : ℕ := 15

/-- The cost per page for additional pages beyond 300 (in tenths of yuan) -/
def cost_additional : ℕ := 10

/-- The threshold number of pages for price change -/
def threshold : ℕ := 300

/-- The ratio of total cost to single set cost -/
def cost_ratio : ℕ := 20

/-- The function to calculate the cost of photocopying a single set of materials -/
def single_set_cost (pages : ℕ) : ℕ :=
  if pages ≤ threshold then
    pages * cost_first_300
  else
    threshold * cost_first_300 + (pages - threshold) * cost_additional

/-- The function to calculate the cost of photocopying all sets of materials -/
def total_cost (pages : ℕ) : ℕ :=
  if team_members * pages ≤ threshold then
    team_members * pages * cost_first_300
  else
    threshold * cost_first_300 + (team_members * pages - threshold) * cost_additional

/-- The theorem stating that 950 pages satisfies the given conditions -/
theorem photocopy_pages_theorem :
  ∃ (pages : ℕ), pages = 950 ∧ total_cost pages = cost_ratio * single_set_cost pages :=
sorry

end NUMINAMATH_CALUDE_photocopy_pages_theorem_l2795_279535


namespace NUMINAMATH_CALUDE_max_distance_between_sine_cosine_curves_l2795_279534

theorem max_distance_between_sine_cosine_curves : 
  ∃ (C : ℝ), C = (Real.sqrt 3 / 2) * Real.sqrt 2 ∧ 
  ∀ (x : ℝ), |Real.sin (x + π/6) - 2 * Real.cos x| ≤ C ∧
  ∃ (a : ℝ), |Real.sin (a + π/6) - 2 * Real.cos a| = C :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_sine_cosine_curves_l2795_279534


namespace NUMINAMATH_CALUDE_softball_team_ratio_l2795_279595

/-- Proves that for a team with 4 more women than men and 20 total players, the ratio of men to women is 2:3 -/
theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
  women = men + 4 →
  men + women = 20 →
  (men : ℚ) / women = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l2795_279595


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l2795_279599

def first_n_even_sum (n : ℕ) : ℕ := n * (n + 1)

def first_n_odd_sum (n : ℕ) : ℕ := n^2

theorem even_odd_sum_difference : 
  first_n_even_sum 1500 - first_n_odd_sum 1500 = 1500 :=
by sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l2795_279599


namespace NUMINAMATH_CALUDE_dog_distance_proof_l2795_279511

/-- The distance the dog runs when Ivan travels from work to home -/
def dog_distance (total_distance : ℝ) : ℝ :=
  2 * total_distance

theorem dog_distance_proof (total_distance : ℝ) (h1 : total_distance = 6) :
  dog_distance total_distance = 12 :=
by
  sorry

#check dog_distance_proof

end NUMINAMATH_CALUDE_dog_distance_proof_l2795_279511


namespace NUMINAMATH_CALUDE_unique_real_solution_l2795_279565

theorem unique_real_solution (b : ℝ) :
  ∀ a : ℝ, (∃! x : ℝ, x^3 - a*x^2 - (2*a + b)*x + a^2 + b = 0) ↔ a < 3 + 4*b :=
by sorry

end NUMINAMATH_CALUDE_unique_real_solution_l2795_279565


namespace NUMINAMATH_CALUDE_choose_three_from_seven_l2795_279543

theorem choose_three_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_seven_l2795_279543


namespace NUMINAMATH_CALUDE_intersection_condition_l2795_279571

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M (a : ℝ) : Set ℝ := {x | x + a ≥ 0}

-- Define set N
def N : Set ℝ := {x | x - 2 < 1}

-- Theorem statement
theorem intersection_condition (a : ℝ) :
  M a ∩ (Set.compl N) = {x | x ≥ 3} → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l2795_279571


namespace NUMINAMATH_CALUDE_parabola_translation_l2795_279530

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk 1 (-6) 5
  let translated := translate original 1 2
  translated = Parabola.mk 1 (-8) 14 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2795_279530


namespace NUMINAMATH_CALUDE_algebraic_division_l2795_279515

theorem algebraic_division (m : ℝ) : -20 * m^6 / (5 * m^2) = -4 * m^4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_division_l2795_279515


namespace NUMINAMATH_CALUDE_inequality_solution_sum_l2795_279598

theorem inequality_solution_sum (m n : ℝ) : 
  (∀ x, x ∈ Set.Ioo m n ↔ (m * x - 1) / (x + 3) > 0) →
  m + n = -10/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sum_l2795_279598


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_three_sqrt_one_third_l2795_279527

theorem sqrt_twelve_minus_three_sqrt_one_third (x : ℝ) : 
  Real.sqrt 12 - 3 * Real.sqrt (1/3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_three_sqrt_one_third_l2795_279527


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l2795_279526

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l2795_279526


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2795_279513

theorem inequality_system_solution :
  let S := {x : ℝ | (x - 1 < 2) ∧ (2*x + 3 ≥ x - 1)}
  S = {x : ℝ | -4 ≤ x ∧ x < 3} :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2795_279513


namespace NUMINAMATH_CALUDE_equation_solution_l2795_279522

theorem equation_solution (n k l m : ℕ) : 
  l > 1 → 
  (1 + n^k)^l = 1 + n^m →
  n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2795_279522


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2795_279553

theorem arithmetic_calculations :
  (58 + 15 * 4 = 118) ∧
  (216 - 72 / 8 = 207) ∧
  ((358 - 295) / 7 = 9) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2795_279553


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2795_279518

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = x^4 := by
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2795_279518


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l2795_279533

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_interior_angles : ℝ := 180 * (n - 2)  -- sum of interior angles formula
  let one_interior_angle : ℝ := sum_interior_angles / n  -- measure of one interior angle
  135

/-- Proof of the theorem -/
lemma prove_regular_octagon_interior_angle : regular_octagon_interior_angle = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l2795_279533


namespace NUMINAMATH_CALUDE_min_sum_mn_l2795_279557

theorem min_sum_mn (m n : ℕ+) (h : m.val * n.val - 2 * m.val - 3 * n.val - 20 = 0) :
  ∃ (p q : ℕ+), p.val * q.val - 2 * p.val - 3 * q.val - 20 = 0 ∧ 
  p.val + q.val = 20 ∧ 
  ∀ (x y : ℕ+), x.val * y.val - 2 * x.val - 3 * y.val - 20 = 0 → x.val + y.val ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_mn_l2795_279557


namespace NUMINAMATH_CALUDE_constrained_optimization_l2795_279512

theorem constrained_optimization (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h1 : 3*x + 5*y + 7*z = 10) (h2 : x + 2*y + 5*z = 6) :
  let w := 2*x - 3*y + 4*z
  ∃ (max_w : ℝ), (∀ x' y' z' : ℝ, x' ≥ 0 → y' ≥ 0 → z' ≥ 0 →
    3*x' + 5*y' + 7*z' = 10 → x' + 2*y' + 5*z' = 6 →
    2*x' - 3*y' + 4*z' ≤ max_w) ∧
  max_w = 3 ∧ w ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_constrained_optimization_l2795_279512


namespace NUMINAMATH_CALUDE_equation_solution_l2795_279561

theorem equation_solution : ∃ x : ℚ, (3 * x + 5 * x = 800 - (4 * x + 6 * x)) ∧ x = 400 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2795_279561


namespace NUMINAMATH_CALUDE_undecagon_diagonals_l2795_279546

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular undecagon has 11 sides -/
def undecagon_sides : ℕ := 11

/-- Theorem: A regular undecagon (11-sided polygon) has 44 diagonals -/
theorem undecagon_diagonals :
  num_diagonals undecagon_sides = 44 := by sorry

end NUMINAMATH_CALUDE_undecagon_diagonals_l2795_279546


namespace NUMINAMATH_CALUDE_renovation_project_dirt_required_l2795_279574

theorem renovation_project_dirt_required (sand cement total : ℚ)
  (h1 : sand = 0.16666666666666666)
  (h2 : cement = 0.16666666666666666)
  (h3 : total = 0.6666666666666666) :
  total - (sand + cement) = 0.3333333333333333 :=
by sorry

end NUMINAMATH_CALUDE_renovation_project_dirt_required_l2795_279574


namespace NUMINAMATH_CALUDE_simplify_expression_l2795_279582

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 9) - (x + 6)*(3*x - 2) = 7*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2795_279582


namespace NUMINAMATH_CALUDE_sum_of_roots_l2795_279503

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 14*a*x + 15*b = 0 ↔ x = c ∨ x = d) →
  (∀ x : ℝ, x^2 - 14*c*x - 15*d = 0 ↔ x = a ∨ x = b) →
  a + b + c + d = 3150 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2795_279503


namespace NUMINAMATH_CALUDE_sum_of_min_values_is_zero_l2795_279556

-- Define the polynomials P and Q
def P (a b x : ℝ) : ℝ := x^2 + a*x + b
def Q (c d x : ℝ) : ℝ := x^2 + c*x + d

-- Define the composition of P and Q
def PQ (a b c d x : ℝ) : ℝ := P a b (Q c d x)
def QP (a b c d x : ℝ) : ℝ := Q c d (P a b x)

-- State the theorem
theorem sum_of_min_values_is_zero 
  (a b c d : ℝ) 
  (h1 : PQ a b c d 1 = 0)
  (h2 : PQ a b c d 3 = 0)
  (h3 : PQ a b c d 5 = 0)
  (h4 : PQ a b c d 7 = 0)
  (h5 : QP a b c d 2 = 0)
  (h6 : QP a b c d 6 = 0)
  (h7 : QP a b c d 10 = 0)
  (h8 : QP a b c d 14 = 0) :
  ∃ (x y : ℝ), P a b x + Q c d y = 0 ∧ 
  (∀ z, P a b z ≥ P a b x) ∧ 
  (∀ w, Q c d w ≥ Q c d y) :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_values_is_zero_l2795_279556
