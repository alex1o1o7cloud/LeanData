import Mathlib

namespace NUMINAMATH_GPT_find_n_for_divisibility_l1145_114518

def digit_sum_odd_positions := 8 + 4 + 5 + 6 -- The sum of the digits in odd positions
def digit_sum_even_positions (n : ℕ) := 5 + n + 2 -- The sum of the digits in even positions

def is_divisible_by_11 (n : ℕ) := (digit_sum_odd_positions - digit_sum_even_positions n) % 11 = 0

theorem find_n_for_divisibility : is_divisible_by_11 5 :=
by
  -- Proof would go here (but according to the instructions, we'll insert a placeholder)
  sorry

end NUMINAMATH_GPT_find_n_for_divisibility_l1145_114518


namespace NUMINAMATH_GPT_area_of_quadrilateral_l1145_114538

theorem area_of_quadrilateral 
  (area_ΔBDF : ℝ) (area_ΔBFE : ℝ) (area_ΔEFC : ℝ) (area_ΔCDF : ℝ) (h₁ : area_ΔBDF = 5)
  (h₂ : area_ΔBFE = 10) (h₃ : area_ΔEFC = 10) (h₄ : area_ΔCDF = 15) :
  (80 - (area_ΔBDF + area_ΔBFE + area_ΔEFC + area_ΔCDF)) = 40 := 
  by sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l1145_114538


namespace NUMINAMATH_GPT_minute_hand_position_l1145_114519

theorem minute_hand_position (t : ℕ) (h_start : t = 2022) :
  let cycle_minutes := 8
  let net_movement_per_cycle := 2
  let full_cycles := t / cycle_minutes
  let remaining_minutes := t % cycle_minutes
  let full_cycles_movement := full_cycles * net_movement_per_cycle
  let extra_movement := if remaining_minutes <= 5 then remaining_minutes else 5 - (remaining_minutes - 5)
  let total_movement := full_cycles_movement + extra_movement
  (total_movement % 60) = 28 :=
by {
  sorry
}

end NUMINAMATH_GPT_minute_hand_position_l1145_114519


namespace NUMINAMATH_GPT_race_problem_l1145_114511

theorem race_problem
  (total_distance : ℕ)
  (A_time : ℕ)
  (B_extra_time : ℕ)
  (A_speed B_speed : ℕ)
  (A_distance B_distance : ℕ)
  (H1 : total_distance = 120)
  (H2 : A_time = 8)
  (H3 : B_extra_time = 7)
  (H4 : A_speed = total_distance / A_time)
  (H5 : B_speed = total_distance / (A_time + B_extra_time))
  (H6 : A_distance = total_distance)
  (H7 : B_distance = B_speed * A_time) :
  A_distance - B_distance = 56 := 
sorry

end NUMINAMATH_GPT_race_problem_l1145_114511


namespace NUMINAMATH_GPT_largest_prime_factor_of_85_l1145_114599

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p

theorem largest_prime_factor_of_85 :
  let a := 65
  let b := 85
  let c := 91
  let d := 143
  let e := 169
  largest_prime_factor b 17 :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_factor_of_85_l1145_114599


namespace NUMINAMATH_GPT_parabola_vertex_range_l1145_114542

def parabola_vertex_in_first_quadrant (m : ℝ) : Prop :=
  ∃ v : ℝ × ℝ, v = (m, m - 1) ∧ 0 < m ∧ 0 < (m - 1)

theorem parabola_vertex_range (m : ℝ) (h_vertex : parabola_vertex_in_first_quadrant m) :
  1 < m :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_range_l1145_114542


namespace NUMINAMATH_GPT_fraction_given_away_is_three_fifths_l1145_114560

variable (initial_bunnies : ℕ) (final_bunnies : ℕ) (kittens_per_bunny : ℕ)

def fraction_given_away (given_away : ℕ) (initial_bunnies : ℕ) : ℚ :=
  given_away / initial_bunnies

theorem fraction_given_away_is_three_fifths 
  (initial_bunnies : ℕ := 30) (final_bunnies : ℕ := 54) (kittens_per_bunny : ℕ := 2)
  (h : final_bunnies = initial_bunnies + kittens_per_bunny * (initial_bunnies - 18)) : 
  fraction_given_away 18 initial_bunnies = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_given_away_is_three_fifths_l1145_114560


namespace NUMINAMATH_GPT_number_of_chickens_l1145_114587

def cost_per_chicken := 3
def total_cost := 15
def potato_cost := 6
def remaining_amount := total_cost - potato_cost

theorem number_of_chickens : (total_cost - potato_cost) / cost_per_chicken = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_chickens_l1145_114587


namespace NUMINAMATH_GPT_tobias_swimming_distance_l1145_114592

def swimming_time_per_100_meters : ℕ := 5
def pause_time : ℕ := 5
def swimming_period : ℕ := 25
def total_visit_hours : ℕ := 3

theorem tobias_swimming_distance :
  let total_visit_minutes := total_visit_hours * 60
  let sequence_time := swimming_period + pause_time
  let number_of_sequences := total_visit_minutes / sequence_time
  let total_pause_time := number_of_sequences * pause_time
  let total_swimming_time := total_visit_minutes - total_pause_time
  let number_of_100m_lengths := total_swimming_time / swimming_time_per_100_meters
  let total_distance := number_of_100m_lengths * 100
  total_distance = 3000 :=
by
  sorry

end NUMINAMATH_GPT_tobias_swimming_distance_l1145_114592


namespace NUMINAMATH_GPT_intersection_dist_general_l1145_114541

theorem intersection_dist_general {a b : ℝ} 
  (h1 : (a^2 + 1) * (a^2 + 4 * (b + 1)) = 34)
  (h2 : (a^2 + 1) * (a^2 + 4 * (b + 2)) = 42) : 
  ∀ x1 x2 : ℝ, 
  x1 ≠ x2 → 
  (x1 * x1 = a * x1 + b - 1 ∧ x2 * x2 = a * x2 + b - 1) → 
  |x2 - x1| = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_dist_general_l1145_114541


namespace NUMINAMATH_GPT_photos_on_last_page_l1145_114529

noncomputable def total_photos : ℕ := 10 * 35 * 4
noncomputable def photos_per_page_after_reorganization : ℕ := 8
noncomputable def total_pages_needed : ℕ := (total_photos + photos_per_page_after_reorganization - 1) / photos_per_page_after_reorganization
noncomputable def pages_filled_in_first_6_albums : ℕ := 6 * 35
noncomputable def last_page_photos : ℕ := if total_pages_needed ≤ pages_filled_in_first_6_albums then 0 else total_photos % photos_per_page_after_reorganization

theorem photos_on_last_page : last_page_photos = 0 :=
by
  sorry

end NUMINAMATH_GPT_photos_on_last_page_l1145_114529


namespace NUMINAMATH_GPT_largest_angle_of_scalene_triangle_l1145_114547

-- Define the problem statement in Lean
theorem largest_angle_of_scalene_triangle (x : ℝ) (hx : x = 30) : 3 * x = 90 :=
by {
  -- Given that the smallest angle is x and x = 30 degrees
  sorry
}

end NUMINAMATH_GPT_largest_angle_of_scalene_triangle_l1145_114547


namespace NUMINAMATH_GPT_max_value_sqrt_expr_l1145_114502

open Real

theorem max_value_sqrt_expr (x y z : ℝ)
  (h1 : x + y + z = 1)
  (h2 : x ≥ -1/3)
  (h3 : y ≥ -1)
  (h4 : z ≥ -5/3) :
  (sqrt (3 * x + 1) + sqrt (3 * y + 3) + sqrt (3 * z + 5)) ≤ 6 :=
  sorry

end NUMINAMATH_GPT_max_value_sqrt_expr_l1145_114502


namespace NUMINAMATH_GPT_total_rocks_is_300_l1145_114509

-- Definitions of rock types in Cliff's collection
variables (I S M : ℕ) -- I: number of igneous rocks, S: number of sedimentary rocks, M: number of metamorphic rocks
variables (shinyI shinyS shinyM : ℕ) -- shinyI: shiny igneous rocks, shinyS: shiny sedimentary rocks, shinyM: shiny metamorphic rocks

-- Given conditions
def igneous_one_third_shiny (I shinyI : ℕ) := 2 * shinyI = 3 * I
def sedimentary_two_ig_as_sed (S I : ℕ) := S = 2 * I
def metamorphic_twice_as_ig (M I : ℕ) := M = 2 * I
def shiny_igneous_is_40 (shinyI : ℕ) := shinyI = 40
def one_fifth_sed_shiny (S shinyS : ℕ) := 5 * shinyS = S
def three_quarters_met_shiny (M shinyM : ℕ) := 4 * shinyM = 3 * M

-- Theorem statement
theorem total_rocks_is_300 (I S M shinyI shinyS shinyM : ℕ)
  (h1 : igneous_one_third_shiny I shinyI)
  (h2 : sedimentary_two_ig_as_sed S I)
  (h3 : metamorphic_twice_as_ig M I)
  (h4 : shiny_igneous_is_40 shinyI)
  (h5 : one_fifth_sed_shiny S shinyS)
  (h6 : three_quarters_met_shiny M shinyM) :
  (I + S + M) = 300 :=
sorry -- Proof to be completed

end NUMINAMATH_GPT_total_rocks_is_300_l1145_114509


namespace NUMINAMATH_GPT_arithmetic_sequence_y_value_l1145_114545

theorem arithmetic_sequence_y_value (y : ℚ) :
  ∃ y : ℚ, 
    (y - 2) - (2/3) = (4 * y - 1) - (y - 2) → 
    y = 11/6 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_y_value_l1145_114545


namespace NUMINAMATH_GPT_countFibSequences_l1145_114593

-- Define what it means for a sequence to be Fibonacci-type
def isFibType (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a n = a (n - 1) + a (n - 2)

-- Define a Fibonacci-type sequence condition with given constraints
def fibSeqCondition (a : ℤ → ℤ) (N : ℤ) : Prop :=
  isFibType a ∧ ∃ n : ℤ, 0 < a n ∧ a n ≤ N ∧ 0 < a (n + 1) ∧ a (n + 1) ≤ N

-- Main theorem
theorem countFibSequences (N : ℤ) :
  ∃ count : ℤ,
    (N % 2 = 0 → count = (N / 2) * (N / 2 + 1)) ∧
    (N % 2 = 1 → count = ((N + 1) / 2) ^ 2) ∧
    (∀ a : ℤ → ℤ, fibSeqCondition a N → (∃ n : ℤ, a n = count)) :=
by
  sorry

end NUMINAMATH_GPT_countFibSequences_l1145_114593


namespace NUMINAMATH_GPT_becky_packs_lunch_days_l1145_114521

-- Definitions of conditions
def school_days := 180
def aliyah_packing_fraction := 1 / 2
def becky_relative_fraction := 1 / 2

-- Derived quantities from conditions
def aliyah_pack_days := school_days * aliyah_packing_fraction
def becky_pack_days := aliyah_pack_days * becky_relative_fraction

-- Statement to prove
theorem becky_packs_lunch_days : becky_pack_days = 45 := by
  sorry

end NUMINAMATH_GPT_becky_packs_lunch_days_l1145_114521


namespace NUMINAMATH_GPT_min_packs_126_l1145_114534

-- Define the sizes of soda packs
def pack_sizes : List ℕ := [6, 12, 24, 48]

-- Define the total number of cans required
def total_cans : ℕ := 126

-- Define a function to calculate the minimum number of packs required
noncomputable def min_packs_to_reach_target (target : ℕ) (sizes : List ℕ) : ℕ :=
sorry -- Implementation will be complex dynamic programming or greedy algorithm

-- The main theorem statement to prove
theorem min_packs_126 (P : ℕ) (h1 : (min_packs_to_reach_target total_cans pack_sizes) = P) : P = 4 :=
sorry -- Proof not required

end NUMINAMATH_GPT_min_packs_126_l1145_114534


namespace NUMINAMATH_GPT_tan_105_degree_is_neg_sqrt3_minus_2_l1145_114577

theorem tan_105_degree_is_neg_sqrt3_minus_2 :
  Real.tan (105 * Real.pi / 180) = -(Real.sqrt 3 + 2) := by
  sorry

end NUMINAMATH_GPT_tan_105_degree_is_neg_sqrt3_minus_2_l1145_114577


namespace NUMINAMATH_GPT_trains_length_difference_eq_zero_l1145_114531

theorem trains_length_difference_eq_zero
  (T1_pole_time : ℕ) (T1_platform_time : ℕ) (T2_pole_time : ℕ) (T2_platform_time : ℕ) (platform_length : ℕ)
  (h1 : T1_pole_time = 11)
  (h2 : T1_platform_time = 22)
  (h3 : T2_pole_time = 15)
  (h4 : T2_platform_time = 30)
  (h5 : platform_length = 120) :
  let L1 := T1_pole_time * platform_length / (T1_platform_time - T1_pole_time)
  let L2 := T2_pole_time * platform_length / (T2_platform_time - T2_pole_time)
  L1 = L2 :=
by
  sorry

end NUMINAMATH_GPT_trains_length_difference_eq_zero_l1145_114531


namespace NUMINAMATH_GPT_wendy_baked_29_cookies_l1145_114517

variables (cupcakes : ℕ) (pastries_taken_home : ℕ) (pastries_sold : ℕ)

def total_initial_pastries (cupcakes pastries_taken_home pastries_sold : ℕ) : ℕ :=
  pastries_taken_home + pastries_sold

def cookies_baked (total_initial_pastries cupcakes : ℕ) : ℕ :=
  total_initial_pastries - cupcakes

theorem wendy_baked_29_cookies :
  cupcakes = 4 →
  pastries_taken_home = 24 →
  pastries_sold = 9 →
  cookies_baked (total_initial_pastries cupcakes pastries_taken_home pastries_sold) cupcakes = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end NUMINAMATH_GPT_wendy_baked_29_cookies_l1145_114517


namespace NUMINAMATH_GPT_P_Ravi_is_02_l1145_114572

def P_Ram : ℚ := 6 / 7
def P_Ram_and_Ravi : ℚ := 0.17142857142857143

theorem P_Ravi_is_02 (P_Ravi : ℚ) : P_Ram_and_Ravi = P_Ram * P_Ravi → P_Ravi = 0.2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_P_Ravi_is_02_l1145_114572


namespace NUMINAMATH_GPT_value_of_product_l1145_114574

theorem value_of_product (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) : (x + 2) * (y + 2) = 16 := by
  sorry

end NUMINAMATH_GPT_value_of_product_l1145_114574


namespace NUMINAMATH_GPT_average_beef_sales_l1145_114514

def ground_beef_sales.Thur : ℕ := 210
def ground_beef_sales.Fri : ℕ := 2 * ground_beef_sales.Thur
def ground_beef_sales.Sat : ℕ := 150
def ground_beef_sales.total : ℕ := ground_beef_sales.Thur + ground_beef_sales.Fri + ground_beef_sales.Sat
def ground_beef_sales.days : ℕ := 3
def ground_beef_sales.average : ℕ := ground_beef_sales.total / ground_beef_sales.days

theorem average_beef_sales (thur : ℕ) (fri : ℕ) (sat : ℕ) (days : ℕ) (total : ℕ) (avg : ℕ) :
  thur = 210 → 
  fri = 2 * thur → 
  sat = 150 → 
  total = thur + fri + sat → 
  days = 3 → 
  avg = total / days → 
  avg = 260 := by
    sorry

end NUMINAMATH_GPT_average_beef_sales_l1145_114514


namespace NUMINAMATH_GPT_column_heights_achievable_l1145_114590

open Int

noncomputable def number_of_column_heights (n : ℕ) (h₁ h₂ h₃ : ℕ) : ℕ :=
  let min_height := n * h₁
  let max_height := n * h₃
  max_height - min_height + 1

theorem column_heights_achievable :
  number_of_column_heights 80 3 8 15 = 961 := by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_column_heights_achievable_l1145_114590


namespace NUMINAMATH_GPT_periodic_even_l1145_114540

noncomputable def f : ℝ → ℝ := sorry  -- We assume the existence of such a function.

variables {α β : ℝ}  -- acute angles of a right triangle

-- Function properties
theorem periodic_even (h_periodic: ∀ x: ℝ, f (x + 2) = f x)
  (h_even: ∀ x: ℝ, f (-x) = f x)
  (h_decreasing: ∀ x y: ℝ, -3 ≤ x ∧ x < y ∧ y ≤ -2 → f x > f y)
  (h_inc_interval_0_1: ∀ x y: ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y)
  (ha: 0 < α ∧ α < π / 2)
  (hb: 0 < β ∧ β < π / 2)
  (h_sum_right_triangle: α + β = π / 2): f (Real.sin α) > f (Real.cos β) :=
sorry

end NUMINAMATH_GPT_periodic_even_l1145_114540


namespace NUMINAMATH_GPT_money_left_is_correct_l1145_114561

noncomputable def total_income : ℝ := 800000
noncomputable def children_pct : ℝ := 0.2
noncomputable def num_children : ℝ := 3
noncomputable def wife_pct : ℝ := 0.3
noncomputable def donation_pct : ℝ := 0.05

noncomputable def remaining_income_after_donations : ℝ := 
  let distributed_to_children := total_income * children_pct * num_children
  let distributed_to_wife := total_income * wife_pct
  let total_distributed := distributed_to_children + distributed_to_wife
  let remaining_after_family := total_income - total_distributed
  let donation := remaining_after_family * donation_pct
  remaining_after_family - donation

theorem money_left_is_correct :
  remaining_income_after_donations = 76000 := 
by 
  sorry

end NUMINAMATH_GPT_money_left_is_correct_l1145_114561


namespace NUMINAMATH_GPT_problem1_l1145_114557

theorem problem1 (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) :
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = -5 / 6 := 
  sorry

end NUMINAMATH_GPT_problem1_l1145_114557


namespace NUMINAMATH_GPT_workers_in_workshop_l1145_114559

theorem workers_in_workshop :
  (∀ (W : ℕ), 8000 * W = 12000 * 7 + 6000 * (W - 7) → W = 21) :=
by
  intro W h
  sorry

end NUMINAMATH_GPT_workers_in_workshop_l1145_114559


namespace NUMINAMATH_GPT_polygons_intersection_l1145_114597

/-- In a square with an area of 5, nine polygons, each with an area of 1, are placed. 
    Prove that some two of them must have an intersection area of at least 1 / 9. -/
theorem polygons_intersection 
  (S : ℝ) (hS : S = 5)
  (n : ℕ) (hn : n = 9)
  (polygons : Fin n → ℝ) (hpolys : ∀ i, polygons i = 1)
  (intersection : Fin n → Fin n → ℝ) : 
  ∃ i j : Fin n, i ≠ j ∧ intersection i j ≥ 1 / 9 := 
sorry

end NUMINAMATH_GPT_polygons_intersection_l1145_114597


namespace NUMINAMATH_GPT_sector_central_angle_l1145_114598

-- The conditions
def r : ℝ := 2
def S : ℝ := 4

-- The question
theorem sector_central_angle : ∃ α : ℝ, |α| = 2 ∧ S = 0.5 * α * r * r :=
by
  sorry

end NUMINAMATH_GPT_sector_central_angle_l1145_114598


namespace NUMINAMATH_GPT_smallest_number_l1145_114533

-- Define the numbers
def A := 5.67823
def B := 5.67833333333 -- repeating decimal
def C := 5.67838383838 -- repeating decimal
def D := 5.67837837837 -- repeating decimal
def E := 5.6783678367  -- repeating decimal

-- The Lean statement to prove that E is the smallest
theorem smallest_number : E < A ∧ E < B ∧ E < C ∧ E < D :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_l1145_114533


namespace NUMINAMATH_GPT_identity_function_l1145_114581

theorem identity_function {f : ℕ → ℕ} (h : ∀ a b : ℕ, 0 < a → 0 < b → a - f b ∣ a * f a - b * f b) :
  ∀ a : ℕ, 0 < a → f a = a :=
by
  sorry

end NUMINAMATH_GPT_identity_function_l1145_114581


namespace NUMINAMATH_GPT_smallest_ninequality_l1145_114576

theorem smallest_ninequality 
  (n : ℕ) 
  (h : ∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≤ 2 ^ (1 - n)) : 
  n = 2 := 
by
  sorry

end NUMINAMATH_GPT_smallest_ninequality_l1145_114576


namespace NUMINAMATH_GPT_rectangle_area_is_140_l1145_114564

noncomputable def area_of_square (a : ℝ) : ℝ := a * a
noncomputable def length_of_rectangle (r : ℝ) : ℝ := (2 / 5) * r
noncomputable def area_of_rectangle (l : ℝ) (b : ℝ) : ℝ := l * b

theorem rectangle_area_is_140 :
  ∃ (a r l b : ℝ), area_of_square a = 1225 ∧ r = a ∧ l = length_of_rectangle r ∧ b = 10 ∧ area_of_rectangle l b = 140 :=
by
  use 35, 35, 14, 10
  simp [area_of_square, length_of_rectangle, area_of_rectangle]
  sorry

end NUMINAMATH_GPT_rectangle_area_is_140_l1145_114564


namespace NUMINAMATH_GPT_rental_cost_equal_mileage_l1145_114578

theorem rental_cost_equal_mileage :
  ∃ m : ℝ, 
    (21.95 + 0.19 * m = 18.95 + 0.21 * m) ∧ 
    m = 150 :=
by
  sorry

end NUMINAMATH_GPT_rental_cost_equal_mileage_l1145_114578


namespace NUMINAMATH_GPT_find_original_cost_price_l1145_114543

variable (C : ℝ)

-- Conditions
def first_discount (C : ℝ) : ℝ := 0.95 * C
def second_discount (C : ℝ) : ℝ := 0.9215 * C
def loss_price (C : ℝ) : ℝ := 0.90 * C
def gain_price_before_tax (C : ℝ) : ℝ := 1.08 * C
def gain_price_after_tax (C : ℝ) : ℝ := 1.20 * C

-- Prove that original cost price is 1800
theorem find_original_cost_price 
  (h1 : first_discount C = loss_price C)
  (h2 : gain_price_after_tax C - loss_price C = 540) : 
  C = 1800 := 
sorry

end NUMINAMATH_GPT_find_original_cost_price_l1145_114543


namespace NUMINAMATH_GPT_solve_system_l1145_114507

theorem solve_system (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : 2 * b - 3 * a = 4) : b = 2 :=
by {
  -- Given the conditions, we need to show that b = 2
  sorry
}

end NUMINAMATH_GPT_solve_system_l1145_114507


namespace NUMINAMATH_GPT_exists_consecutive_with_square_factors_l1145_114525

theorem exists_consecutive_with_square_factors (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ m : ℕ, m^2 ∣ (k + i) ∧ m > 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_consecutive_with_square_factors_l1145_114525


namespace NUMINAMATH_GPT_intersection_S_T_l1145_114513

def S : Set ℤ := {-4, -3, 6, 7}
def T : Set ℤ := {x | x^2 > 4 * x}

theorem intersection_S_T : S ∩ T = {-4, -3, 6, 7} :=
by
  sorry

end NUMINAMATH_GPT_intersection_S_T_l1145_114513


namespace NUMINAMATH_GPT_ball_radius_l1145_114500

theorem ball_radius (x r : ℝ) 
  (h1 : (15 : ℝ) ^ 2 + x ^ 2 = r ^ 2) 
  (h2 : x + 12 = r) : 
  r = 15.375 := 
sorry

end NUMINAMATH_GPT_ball_radius_l1145_114500


namespace NUMINAMATH_GPT_factorial_expression_l1145_114580

theorem factorial_expression :
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_GPT_factorial_expression_l1145_114580


namespace NUMINAMATH_GPT_base_7_is_good_number_l1145_114524

def is_good_number (m: ℕ) : Prop :=
  ∃ (p: ℕ) (n: ℕ), Prime p ∧ n ≥ 2 ∧ m = p^n

theorem base_7_is_good_number : 
  ∀ b: ℕ, (is_good_number (b^2 - (2 * b + 3))) → b = 7 :=
by
  intro b h
  sorry

end NUMINAMATH_GPT_base_7_is_good_number_l1145_114524


namespace NUMINAMATH_GPT_cubic_roots_number_l1145_114546

noncomputable def determinant_cubic (a b c d : ℝ) (x : ℝ) : ℝ :=
  x * (x^2 + a^2) + c * (b * x + a * b) - b * (c * a - b * x)

theorem cubic_roots_number (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ roots : ℕ, (roots = 1 ∨ roots = 3) :=
  sorry

end NUMINAMATH_GPT_cubic_roots_number_l1145_114546


namespace NUMINAMATH_GPT_speed_second_hour_l1145_114566

noncomputable def speed_in_first_hour : ℝ := 95
noncomputable def average_speed : ℝ := 77.5
noncomputable def total_time : ℝ := 2
def speed_in_second_hour : ℝ := sorry -- to be deduced

theorem speed_second_hour :
  speed_in_second_hour = 60 :=
by
  sorry

end NUMINAMATH_GPT_speed_second_hour_l1145_114566


namespace NUMINAMATH_GPT_find_value_of_expression_l1145_114570

noncomputable def roots_g : Set ℂ := { x | x^2 - 3*x - 2 = 0 }

theorem find_value_of_expression:
  ∀ γ δ : ℂ, γ ∈ roots_g → δ ∈ roots_g →
  (γ + δ = 3) → (7 * γ^4 + 10 * δ^3 = 1363) :=
by
  intros γ δ hγ hδ hsum
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l1145_114570


namespace NUMINAMATH_GPT_water_on_wednesday_l1145_114582

-- Define the total water intake for the week.
def total_water : ℕ := 60

-- Define the water intake amounts for specific days.
def water_on_mon_thu_sat : ℕ := 9
def water_on_tue_fri_sun : ℕ := 8

-- Define the number of days for each intake.
def days_mon_thu_sat : ℕ := 3
def days_tue_fri_sun : ℕ := 3

-- Define the water intake calculated for specific groups of days.
def total_water_mon_thu_sat := water_on_mon_thu_sat * days_mon_thu_sat
def total_water_tue_fri_sun := water_on_tue_fri_sun * days_tue_fri_sun

-- Define the total water intake for these days combined.
def total_water_other_days := total_water_mon_thu_sat + total_water_tue_fri_sun

-- Define the water intake for Wednesday, which we need to prove to be 9 liters.
theorem water_on_wednesday : total_water - total_water_other_days = 9 := by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_water_on_wednesday_l1145_114582


namespace NUMINAMATH_GPT_hermans_breakfast_cost_l1145_114516

-- Define the conditions
def meals_per_day : Nat := 4
def days_per_week : Nat := 5
def cost_per_meal : Nat := 4
def total_weeks : Nat := 16

-- Define the statement to prove
theorem hermans_breakfast_cost :
  (meals_per_day * days_per_week * cost_per_meal * total_weeks) = 1280 := by
  sorry

end NUMINAMATH_GPT_hermans_breakfast_cost_l1145_114516


namespace NUMINAMATH_GPT_find_DG_l1145_114554

theorem find_DG 
  (a b : ℕ) -- sides DE and EC
  (S : ℕ := 19 * (a + b)) -- area of each rectangle
  (k l : ℕ) -- sides DG and CH
  (h1 : S = a * k) 
  (h2 : S = b * l) 
  (h_bc : 19 * (a + b) = S)
  (h_div_a : S % a = 0)
  (h_div_b : S % b = 0)
  : DG = 380 :=
sorry

end NUMINAMATH_GPT_find_DG_l1145_114554


namespace NUMINAMATH_GPT_range_of_a_l1145_114579

variable {α : Type} [LinearOrderedField α]

def A (a : α) : Set α := {x | |x - a| ≤ 1}

def B : Set α := {x | x^2 - 5*x + 4 ≥ 0}

theorem range_of_a (a : α) (h : A a ∩ B = ∅) : 2 < a ∧ a < 3 := sorry

end NUMINAMATH_GPT_range_of_a_l1145_114579


namespace NUMINAMATH_GPT_parabola_equation_l1145_114553

theorem parabola_equation :
  (∃ h k : ℝ, h^2 = 3 ∧ k^2 = 6) →
  (∃ c : ℝ, c^2 = (3 + 6)) →
  (∃ x y : ℝ, x = 3 ∧ y = 0) →
  (y^2 = 12 * x) :=
sorry

end NUMINAMATH_GPT_parabola_equation_l1145_114553


namespace NUMINAMATH_GPT_lift_time_15_minutes_l1145_114562

theorem lift_time_15_minutes (t : ℕ) (h₁ : 5 = 5) (h₂ : 6 * (t + 5) = 120) : t = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_lift_time_15_minutes_l1145_114562


namespace NUMINAMATH_GPT_smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l1145_114583

theorem smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5 :
  ∃ n : ℕ, n > 0 ∧ (7^n % 5 = n^7 % 5) ∧
  ∀ m : ℕ, m > 0 ∧ (7^m % 5 = m^7 % 5) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l1145_114583


namespace NUMINAMATH_GPT_altered_prism_edges_l1145_114575

theorem altered_prism_edges :
  let original_edges := 12
  let vertices := 8
  let edges_per_vertex := 3
  let faces := 6
  let edges_per_face := 1
  let total_edges := original_edges + edges_per_vertex * vertices + edges_per_face * faces
  total_edges = 42 :=
by
  let original_edges := 12
  let vertices := 8
  let edges_per_vertex := 3
  let faces := 6
  let edges_per_face := 1
  let total_edges := original_edges + edges_per_vertex * vertices + edges_per_face * faces
  show total_edges = 42
  sorry

end NUMINAMATH_GPT_altered_prism_edges_l1145_114575


namespace NUMINAMATH_GPT_hyperbola_through_C_l1145_114558

noncomputable def equation_of_hyperbola_passing_through_C : Prop :=
  let A := (-1/2, 1/4)
  let B := (2, 4)
  let C := (-1/2, 4)
  ∃ (k : ℝ), k = -2 ∧ (∀ x : ℝ, x ≠ 0 → x * (4) = k)

theorem hyperbola_through_C :
  equation_of_hyperbola_passing_through_C :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_through_C_l1145_114558


namespace NUMINAMATH_GPT_number_divided_by_005_l1145_114505

theorem number_divided_by_005 (number : ℝ) (h : number / 0.05 = 1500) : number = 75 :=
sorry

end NUMINAMATH_GPT_number_divided_by_005_l1145_114505


namespace NUMINAMATH_GPT_geometric_sequence_general_term_l1145_114515

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ q : ℝ, q > 0 ∧ (∀ n, a (n + 1) = q * a n)

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h_seq : geometric_sequence a) 
  (h_S3 : a 1 * (1 + (a 2 / a 1) + (a 3 / a 1)) = 21) 
  (h_condition : 2 * a 2 = a 3) :
  ∃ c : ℝ, c = 3 ∧ ∀ n, a n = 3 * 2^(n - 1) := sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_l1145_114515


namespace NUMINAMATH_GPT_matrix_det_eq_seven_l1145_114586

theorem matrix_det_eq_seven (p q r s : ℝ) (h : p * s - q * r = 7) : 
  (p - 2 * r) * s - (q - 2 * s) * r = 7 := 
sorry

end NUMINAMATH_GPT_matrix_det_eq_seven_l1145_114586


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l1145_114591

theorem average_of_remaining_two_numbers 
(A B C D E F G H : ℝ) 
(h_avg1 : (A + B + C + D + E + F + G + H) / 8 = 4.5) 
(h_avg2 : (A + B + C) / 3 = 5.2) 
(h_avg3 : (D + E + F) / 3 = 3.6) : 
  ((G + H) / 2 = 4.8) :=
sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l1145_114591


namespace NUMINAMATH_GPT_train_pass_time_l1145_114508

-- Assuming conversion factor, length of the train, and speed in km/hr
def conversion_factor := 1000 / 3600
def train_length := 280
def speed_km_hr := 36

-- Defining speed in m/s
def speed_m_s := speed_km_hr * conversion_factor

-- Defining the time to pass a tree
def time_to_pass_tree := train_length / speed_m_s

-- Theorem statement
theorem train_pass_time : time_to_pass_tree = 28 := by
  sorry

end NUMINAMATH_GPT_train_pass_time_l1145_114508


namespace NUMINAMATH_GPT_cosine_periodicity_l1145_114506

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end NUMINAMATH_GPT_cosine_periodicity_l1145_114506


namespace NUMINAMATH_GPT_hyperbola_parameters_sum_l1145_114520

theorem hyperbola_parameters_sum :
  ∃ (h k a b : ℝ), 
    (h = 2 ∧ k = 0 ∧ a = 3 ∧ b = 3 * Real.sqrt 3) ∧
    h + k + a + b = 3 * Real.sqrt 3 + 5 := by
  sorry

end NUMINAMATH_GPT_hyperbola_parameters_sum_l1145_114520


namespace NUMINAMATH_GPT_simplify_expression_l1145_114589

theorem simplify_expression : (1 / (1 / ((1 / 3) ^ 1) + 1 / ((1 / 3) ^ 2) + 1 / ((1 / 3) ^ 3))) = (1 / 39) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1145_114589


namespace NUMINAMATH_GPT_Diamond_result_l1145_114571

-- Define the binary operation Diamond
def Diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem Diamond_result : Diamond (Diamond 3 4) 2 = 179 := 
by 
  sorry

end NUMINAMATH_GPT_Diamond_result_l1145_114571


namespace NUMINAMATH_GPT_number_of_cows_l1145_114537

theorem number_of_cows (x y : ℕ) 
  (h1 : 4 * x + 2 * y = 14 + 2 * (x + y)) : 
  x = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cows_l1145_114537


namespace NUMINAMATH_GPT_samantha_more_posters_l1145_114594

theorem samantha_more_posters :
  ∃ S : ℕ, S > 18 ∧ 18 + S = 51 ∧ S - 18 = 15 :=
by
  sorry

end NUMINAMATH_GPT_samantha_more_posters_l1145_114594


namespace NUMINAMATH_GPT_shaded_area_l1145_114512

theorem shaded_area (R : ℝ) (π : ℝ) (h1 : π * (R / 2)^2 * 2 = 1) : 
  (π * R^2 - (π * (R / 2)^2 * 2)) = 1 := 
by
  sorry

end NUMINAMATH_GPT_shaded_area_l1145_114512


namespace NUMINAMATH_GPT_geometric_sequence_not_sufficient_nor_necessary_l1145_114539

theorem geometric_sequence_not_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) → 
  (¬ (q > 1 → ∀ n : ℕ, a n < a (n + 1))) ∧ (¬ (∀ n : ℕ, a n < a (n + 1) → q > 1)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_not_sufficient_nor_necessary_l1145_114539


namespace NUMINAMATH_GPT_dad_steps_eq_90_l1145_114573

-- Define the conditions given in the problem
variables (masha_steps yasha_steps dad_steps : ℕ)

-- Conditions:
-- 1. Dad takes 3 steps while Masha takes 5 steps
-- 2. Masha takes 3 steps while Yasha takes 5 steps
-- 3. Together, Masha and Yasha made 400 steps
def conditions := dad_steps * 5 = 3 * masha_steps ∧ masha_steps * yasha_steps = 3 * yasha_steps ∧ 3 * yasha_steps = 400

-- Theorem stating the proof problem
theorem dad_steps_eq_90 : conditions masha_steps yasha_steps dad_steps → dad_steps = 90 :=
by
  sorry

end NUMINAMATH_GPT_dad_steps_eq_90_l1145_114573


namespace NUMINAMATH_GPT_roots_modulus_less_than_one_l1145_114528

theorem roots_modulus_less_than_one
  (A B C D : ℝ)
  (h1 : ∀ x, x^2 + A * x + B = 0 → |x| < 1)
  (h2 : ∀ x, x^2 + C * x + D = 0 → |x| < 1) :
  ∀ x, x^2 + (A + C) / 2 * x + (B + D) / 2 = 0 → |x| < 1 :=
by
  sorry

end NUMINAMATH_GPT_roots_modulus_less_than_one_l1145_114528


namespace NUMINAMATH_GPT_eq_frac_l1145_114568

noncomputable def g : ℝ → ℝ := sorry

theorem eq_frac (h1 : ∀ c d : ℝ, c^3 * g d = d^3 * g c)
                (h2 : g 3 ≠ 0) : (g 7 - g 4) / g 3 = 279 / 27 :=
by
  sorry

end NUMINAMATH_GPT_eq_frac_l1145_114568


namespace NUMINAMATH_GPT_parabola_right_shift_unique_intersection_parabola_down_shift_unique_intersection_l1145_114549

theorem parabola_right_shift_unique_intersection (p : ℚ) :
  let y := 2 * (x - p)^2;
  (x * x - 4) = 0 →
  p = 31 / 8 := sorry

theorem parabola_down_shift_unique_intersection (q : ℚ) :
  let y := 2 * x^2 - q;
  (x * x - 4) = 0 →
  q = 31 / 8 := sorry

end NUMINAMATH_GPT_parabola_right_shift_unique_intersection_parabola_down_shift_unique_intersection_l1145_114549


namespace NUMINAMATH_GPT_range_of_a_l1145_114536

theorem range_of_a 
  (x y a : ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy : x + y + 3 = x * y) 
  (h_a : ∀ x y : ℝ, (x + y)^2 - a * (x + y) + 1 ≥ 0) :
  a ≤ 37 / 6 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1145_114536


namespace NUMINAMATH_GPT_prob1_prob2_prob3_prob4_l1145_114530

theorem prob1 : (3^3)^2 = 3^6 := by
  sorry

theorem prob2 : (-4 * x * y^3) * (-2 * x^2) = 8 * x^3 * y^3 := by
  sorry

theorem prob3 : 2 * x * (3 * y - x^2) + 2 * x * x^2 = 6 * x * y := by
  sorry

theorem prob4 : (20 * x^3 * y^5 - 10 * x^4 * y^4 - 20 * x^3 * y^2) / (-5 * x^3 * y^2) = -4 * y^3 + 2 * x * y^2 + 4 := by
  sorry

end NUMINAMATH_GPT_prob1_prob2_prob3_prob4_l1145_114530


namespace NUMINAMATH_GPT_division_result_l1145_114569

-- Define n in terms of the given condition
def n : ℕ := 9^2023

theorem division_result : n / 3 = 3^4045 :=
by
  sorry

end NUMINAMATH_GPT_division_result_l1145_114569


namespace NUMINAMATH_GPT_find_coefficients_l1145_114552

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

noncomputable def h (a b c x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_coefficients :
  ∃ a b c : ℝ, (∀ s : ℝ, f s = 0 → h a b c (s^3) = 0) ∧
    (a, b, c) = (-6, -9, 20) :=
sorry

end NUMINAMATH_GPT_find_coefficients_l1145_114552


namespace NUMINAMATH_GPT_dadAgeWhenXiaoHongIs7_l1145_114596

variable {a : ℕ}

-- Condition: Dad's age is given as 'a'
-- Condition: Dad's age is 4 times plus 3 years more than Xiao Hong's age
def xiaoHongAge (a : ℕ) : ℕ := (a - 3) / 4

theorem dadAgeWhenXiaoHongIs7 : xiaoHongAge a = 7 → a = 31 := by
  intro h
  have h1 : a - 3 = 28 := by sorry   -- Algebraic manipulation needed
  have h2 : a = 31 := by sorry       -- Algebraic manipulation needed
  exact h2

end NUMINAMATH_GPT_dadAgeWhenXiaoHongIs7_l1145_114596


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1145_114563

theorem quadratic_no_real_roots (a b : ℝ) (h : ∃ x : ℝ, x^2 + b * x + a = 0) : false :=
sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1145_114563


namespace NUMINAMATH_GPT_candy_total_cost_l1145_114567

theorem candy_total_cost
    (grape_candies cherry_candies apple_candies : ℕ)
    (cost_per_candy : ℝ)
    (h1 : grape_candies = 3 * cherry_candies)
    (h2 : apple_candies = 2 * grape_candies)
    (h3 : cost_per_candy = 2.50)
    (h4 : grape_candies = 24) :
    (grape_candies + cherry_candies + apple_candies) * cost_per_candy = 200 := 
by
  sorry

end NUMINAMATH_GPT_candy_total_cost_l1145_114567


namespace NUMINAMATH_GPT_find_other_root_l1145_114555

theorem find_other_root (m : ℝ) (h : 2^2 - 2 + m = 0) : 
  ∃ α : ℝ, α = -1 ∧ (α^2 - α + m = 0) :=
by
  -- Assuming x = 2 is a root, prove that the other root is -1.
  sorry

end NUMINAMATH_GPT_find_other_root_l1145_114555


namespace NUMINAMATH_GPT_find_n_l1145_114595

theorem find_n (n : ℕ) (x y a b : ℕ) (hx : x = 1) (hy : y = 1) (ha : a = 1) (hb : b = 1)
  (h : (x + 3 * y) ^ n = (7 * a + b) ^ 10) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1145_114595


namespace NUMINAMATH_GPT_pencils_in_boxes_l1145_114588

theorem pencils_in_boxes (total_pencils : ℕ) (pencils_per_box : ℕ) (boxes_required : ℕ) 
    (h1 : total_pencils = 648) (h2 : pencils_per_box = 4) : boxes_required = 162 :=
sorry

end NUMINAMATH_GPT_pencils_in_boxes_l1145_114588


namespace NUMINAMATH_GPT_solve_system_l1145_114510

theorem solve_system (x y : ℚ) (h1 : 6 * x = -9 - 3 * y) (h2 : 4 * x = 5 * y - 34) : x = 1/2 ∧ y = -4 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1145_114510


namespace NUMINAMATH_GPT_part_I_solution_part_II_solution_l1145_114556

def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

theorem part_I_solution (x : ℝ) :
  (|x + 1| - |x - 1| >= x) ↔ (x <= -2 ∨ (0 <= x ∧ x <= 2)) :=
by
  sorry

theorem part_II_solution (m : ℝ) :
  (∀ (x a : ℝ), (0 < m ∧ m < 1 ∧ (a <= -3 ∨ 3 <= a)) → (f x a m >= 2)) ↔ (m = 1/3) :=
by
  sorry

end NUMINAMATH_GPT_part_I_solution_part_II_solution_l1145_114556


namespace NUMINAMATH_GPT_cylinder_volume_transformation_l1145_114501

variable (r h : ℝ)
variable (V_original : ℝ)
variable (V_new : ℝ)

noncomputable def original_volume : ℝ := Real.pi * r^2 * h

noncomputable def new_volume : ℝ := Real.pi * (3 * r)^2 * (2 * h)

theorem cylinder_volume_transformation 
  (h_original : original_volume r h = 15) :
  new_volume r h = 270 :=
by
  unfold original_volume at h_original
  unfold new_volume
  sorry

end NUMINAMATH_GPT_cylinder_volume_transformation_l1145_114501


namespace NUMINAMATH_GPT_part1_part2_l1145_114544

theorem part1 (a : ℝ) (ha : z = Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6))
  (h_imag : z.re = 0) : a = 1 :=
sorry

theorem part2 (a : ℝ) (ha : z = Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6))
  (h4thQuad : z.re > 0 ∧ z.im < 0) : -1 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1145_114544


namespace NUMINAMATH_GPT_largest_n_for_factorable_poly_l1145_114503

theorem largest_n_for_factorable_poly :
  ∃ n : ℤ, (∀ A B : ℤ, (3 * B + A = n) ∧ (A * B = 72) → (A = 1 ∧ B = 72 ∧ n = 217)) ∧
           (∀ A B : ℤ, A * B = 72 → 3 * B + A ≤ 217) :=
by
  sorry

end NUMINAMATH_GPT_largest_n_for_factorable_poly_l1145_114503


namespace NUMINAMATH_GPT_find_k_inverse_proportion_l1145_114532

theorem find_k_inverse_proportion :
  ∃ k : ℝ, (∀ x y : ℝ, y = (k + 1) / x → (x = 1 ∧ y = -2) → k = -3) :=
by
  sorry

end NUMINAMATH_GPT_find_k_inverse_proportion_l1145_114532


namespace NUMINAMATH_GPT_marble_probability_correct_l1145_114522

noncomputable def marble_probability : ℚ :=
  let total_ways := (Nat.choose 20 4 : ℚ)
  let ways_two_red := (Nat.choose 12 2 : ℚ)
  let ways_two_blue := (Nat.choose 8 2 : ℚ)
  (ways_two_red * ways_two_blue) / total_ways

theorem marble_probability_correct : marble_probability = 56 / 147 :=
by
  -- Note: the proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_marble_probability_correct_l1145_114522


namespace NUMINAMATH_GPT_mr_william_land_percentage_l1145_114504

def total_tax_collected : ℝ := 3840
def mr_william_tax_paid : ℝ := 480
def expected_percentage : ℝ := 12.5

theorem mr_william_land_percentage :
  (mr_william_tax_paid / total_tax_collected) * 100 = expected_percentage := 
sorry

end NUMINAMATH_GPT_mr_william_land_percentage_l1145_114504


namespace NUMINAMATH_GPT_intersect_is_one_l1145_114584

def SetA : Set ℝ := {x | 0 < x ∧ x < 2}

def SetB : Set ℝ := {0, 1, 2, 3}

theorem intersect_is_one : SetA ∩ SetB = {1} :=
by
  sorry

end NUMINAMATH_GPT_intersect_is_one_l1145_114584


namespace NUMINAMATH_GPT_interval_solution_l1145_114548

theorem interval_solution (x : ℝ) : 
  (1 < 5 * x ∧ 5 * x < 3) ∧ (2 < 8 * x ∧ 8 * x < 4) ↔ (1/4 < x ∧ x < 1/2) := 
by
  sorry

end NUMINAMATH_GPT_interval_solution_l1145_114548


namespace NUMINAMATH_GPT_sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5_l1145_114523

noncomputable def compare_sq_roots_sum : Prop := 
  (Real.sqrt 11 + Real.sqrt 3) < (Real.sqrt 9 + Real.sqrt 5)

theorem sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5 :
  compare_sq_roots_sum :=
sorry

end NUMINAMATH_GPT_sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5_l1145_114523


namespace NUMINAMATH_GPT_sausage_more_than_pepperoni_l1145_114535

noncomputable def pieces_of_meat_per_slice : ℕ := 22
noncomputable def slices : ℕ := 6
noncomputable def total_pieces_of_meat : ℕ := pieces_of_meat_per_slice * slices

noncomputable def pieces_of_pepperoni : ℕ := 30
noncomputable def pieces_of_ham : ℕ := 2 * pieces_of_pepperoni

noncomputable def total_pieces_of_meat_without_sausage : ℕ := pieces_of_pepperoni + pieces_of_ham
noncomputable def pieces_of_sausage : ℕ := total_pieces_of_meat - total_pieces_of_meat_without_sausage

theorem sausage_more_than_pepperoni : (pieces_of_sausage - pieces_of_pepperoni) = 12 := by
  sorry

end NUMINAMATH_GPT_sausage_more_than_pepperoni_l1145_114535


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1145_114527

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1145_114527


namespace NUMINAMATH_GPT_no_prime_divisible_by_42_l1145_114526

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → ¬ (42 ∣ p) :=
by sorry

end NUMINAMATH_GPT_no_prime_divisible_by_42_l1145_114526


namespace NUMINAMATH_GPT_basketball_court_width_l1145_114565

variable (width length : ℕ)

-- Given conditions
axiom h1 : length = width + 14
axiom h2 : 2 * length + 2 * width = 96

-- Prove the width is 17 meters
theorem basketball_court_width : width = 17 :=
by {
  sorry
}

end NUMINAMATH_GPT_basketball_court_width_l1145_114565


namespace NUMINAMATH_GPT_cylinder_height_percentage_l1145_114585

-- Lean 4 statement for the problem
theorem cylinder_height_percentage (h : ℝ) (r : ℝ) (H : ℝ) :
  (7 / 8) * h = (3 / 5) * (1.25 * r)^2 * H → H = 0.9333 * h :=
by 
  sorry

end NUMINAMATH_GPT_cylinder_height_percentage_l1145_114585


namespace NUMINAMATH_GPT_problem_solution_l1145_114551

noncomputable def quadratic_symmetric_b (a : ℝ) : ℝ :=
  2 * (1 - a)

theorem problem_solution (a : ℝ) (h1 : quadratic_symmetric_b a = 6) :
  b = 6 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1145_114551


namespace NUMINAMATH_GPT_smallest_sum_of_exterior_angles_l1145_114550

open Real

theorem smallest_sum_of_exterior_angles 
  (p q r : ℕ) 
  (hp : p > 2) 
  (hq : q > 2) 
  (hr : r > 2) 
  (hpq : p ≠ q) 
  (hqr : q ≠ r) 
  (hrp : r ≠ p) 
  : (360 / p + 360 / q + 360 / r) ≥ 282 ∧ 
    (360 / p + 360 / q + 360 / r) = 282 → 
    360 / p = 120 ∧ 360 / q = 90 ∧ 360 / r = 72 := 
sorry

end NUMINAMATH_GPT_smallest_sum_of_exterior_angles_l1145_114550
