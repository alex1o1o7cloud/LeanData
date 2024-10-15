import Mathlib

namespace NUMINAMATH_CALUDE_odell_kershaw_passing_count_l235_23553

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- radius of the lane in meters
  direction : ℝ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (duration : ℝ) : ℕ :=
  sorry

theorem odell_kershaw_passing_count :
  let odell : Runner := ⟨260, 55, 1⟩
  let kershaw : Runner := ⟨280, 65, -1⟩
  passingCount odell kershaw 45 = 64 :=
sorry

end NUMINAMATH_CALUDE_odell_kershaw_passing_count_l235_23553


namespace NUMINAMATH_CALUDE_sleep_deficit_l235_23595

def weeknights : ℕ := 5
def weekendNights : ℕ := 2
def actualWeekdaySleep : ℕ := 5
def actualWeekendSleep : ℕ := 6
def idealSleep : ℕ := 8

theorem sleep_deficit :
  (weeknights * idealSleep + weekendNights * idealSleep) -
  (weeknights * actualWeekdaySleep + weekendNights * actualWeekendSleep) = 19 := by
  sorry

end NUMINAMATH_CALUDE_sleep_deficit_l235_23595


namespace NUMINAMATH_CALUDE_pencil_cost_to_selling_ratio_l235_23571

/-- The ratio of the cost to the selling price for 70 pencils -/
theorem pencil_cost_to_selling_ratio 
  (C : ℝ) -- Cost price of one pencil
  (S : ℝ) -- Selling price of one pencil
  (h1 : C > 0) -- Assumption that cost is positive
  (h2 : S > 0) -- Assumption that selling price is positive
  (h3 : C > (2/7) * S) -- Assumption that cost is greater than 2/7 of selling price
  : (70 * C) / (70 * C - 20 * S) = C / (C - 2 * S / 7) :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_to_selling_ratio_l235_23571


namespace NUMINAMATH_CALUDE_refrigerator_profit_percentage_l235_23566

/-- Calculates the percentage of profit for a refrigerator sale --/
theorem refrigerator_profit_percentage
  (discounted_price : ℝ)
  (discount_percentage : ℝ)
  (transport_cost : ℝ)
  (installation_cost : ℝ)
  (selling_price : ℝ)
  (h1 : discounted_price = 14500)
  (h2 : discount_percentage = 20)
  (h3 : transport_cost = 125)
  (h4 : installation_cost = 250)
  (h5 : selling_price = 20350) :
  ∃ (profit_percentage : ℝ),
    abs (profit_percentage - 36.81) < 0.01 ∧
    profit_percentage = (selling_price - (discounted_price + transport_cost + installation_cost)) /
                        (discounted_price + transport_cost + installation_cost) * 100 :=
by sorry


end NUMINAMATH_CALUDE_refrigerator_profit_percentage_l235_23566


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_l235_23593

-- Define the quadratic polynomials
def quadratic_poly (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution sets
def solution_set (a b c : ℝ) : Set ℝ := {x | quadratic_poly a b c x > 0}

-- Define the condition for equal ratios
def equal_ratios (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂

-- State the theorem
theorem not_necessary_not_sufficient
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  ¬(equal_ratios a₁ b₁ c₁ a₂ b₂ c₂ ↔ solution_set a₁ b₁ c₁ = solution_set a₂ b₂ c₂) :=
sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_l235_23593


namespace NUMINAMATH_CALUDE_circle_circumference_area_relation_l235_23536

theorem circle_circumference_area_relation : 
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 12 → π * d < 10 * (π * d^2 / 4) := by
  sorry

#check circle_circumference_area_relation

end NUMINAMATH_CALUDE_circle_circumference_area_relation_l235_23536


namespace NUMINAMATH_CALUDE_line_perp_from_plane_perp_l235_23577

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perpLine : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perpPlane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpLineLine : Line → Line → Prop)

-- Theorem statement
theorem line_perp_from_plane_perp 
  (a b : Line) (α β : Plane) 
  (h1 : perpLine a α) 
  (h2 : perpLine b β) 
  (h3 : perpPlane α β) : 
  perpLineLine a b :=
sorry

end NUMINAMATH_CALUDE_line_perp_from_plane_perp_l235_23577


namespace NUMINAMATH_CALUDE_speaking_orders_eq_720_l235_23527

/-- The number of different speaking orders for selecting 4 students from a group of 7 students,
    including students A and B, with the requirement that at least one of A or B must participate. -/
def speakingOrders : ℕ :=
  Nat.descFactorial 7 4 - Nat.descFactorial 5 4

/-- Theorem stating that the number of speaking orders is 720. -/
theorem speaking_orders_eq_720 : speakingOrders = 720 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_eq_720_l235_23527


namespace NUMINAMATH_CALUDE_power_inequality_l235_23568

theorem power_inequality (a b n : ℕ) (ha : a > b) (hb : b > 1) (hodd : Odd b) 
  (hn : n > 0) (hdiv : (b^n : ℕ) ∣ (a^n - 1)) : 
  (a : ℝ)^b > (3 : ℝ)^n / n := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l235_23568


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l235_23558

theorem gasoline_price_increase 
  (spending_increase : Real) 
  (quantity_decrease : Real) 
  (price_increase : Real) : 
  spending_increase = 0.15 → 
  quantity_decrease = 0.08000000000000007 → 
  (1 + price_increase) * (1 - quantity_decrease) = 1 + spending_increase → 
  price_increase = 0.25 := by
sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l235_23558


namespace NUMINAMATH_CALUDE_no_solution_exists_l235_23587

theorem no_solution_exists : ¬∃ x : ℝ, (x / (-4) ≥ 3 + x) ∧ (|2 * x - 1| < 4 + 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l235_23587


namespace NUMINAMATH_CALUDE_three_fourths_of_45_l235_23534

theorem three_fourths_of_45 : (3 : ℚ) / 4 * 45 = 33 + 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_of_45_l235_23534


namespace NUMINAMATH_CALUDE_point_not_on_line_l235_23597

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) : 
  ¬(∃ y : ℝ, y = 3 * m * 4 + 4 * b ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_point_not_on_line_l235_23597


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l235_23545

theorem price_decrease_percentage (P : ℝ) (x : ℝ) (h₁ : P > 0) :
  (1.20 * P) * (1 - x / 100) = 0.75 * P → x = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l235_23545


namespace NUMINAMATH_CALUDE_basketball_league_female_fraction_l235_23544

theorem basketball_league_female_fraction :
  -- Define variables
  let male_last_year : ℕ := 30
  let total_increase_rate : ℚ := 115 / 100
  let male_increase_rate : ℚ := 110 / 100
  let female_increase_rate : ℚ := 125 / 100

  -- Calculate values
  let male_this_year : ℚ := male_last_year * male_increase_rate
  let female_last_year : ℚ := (total_increase_rate * (male_last_year : ℚ) - male_this_year) / (female_increase_rate - total_increase_rate)
  let female_this_year : ℚ := female_last_year * female_increase_rate
  let total_this_year : ℚ := male_this_year + female_this_year

  -- Prove the fraction
  female_this_year / total_this_year = 25 / 69 := by sorry

end NUMINAMATH_CALUDE_basketball_league_female_fraction_l235_23544


namespace NUMINAMATH_CALUDE_sold_below_cost_price_l235_23542

def cost_price : ℚ := 5625
def profit_percentage : ℚ := 16 / 100
def additional_amount : ℚ := 1800

def selling_price_with_profit : ℚ := cost_price * (1 + profit_percentage)
def actual_selling_price : ℚ := selling_price_with_profit - additional_amount

def percentage_below_cost : ℚ := (cost_price - actual_selling_price) / cost_price * 100

theorem sold_below_cost_price : percentage_below_cost = 16 := by sorry

end NUMINAMATH_CALUDE_sold_below_cost_price_l235_23542


namespace NUMINAMATH_CALUDE_no_real_roots_l235_23578

theorem no_real_roots : ∀ x : ℝ, 2 * Real.cos (x / 2) ≠ 10^x + 10^(-x) + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l235_23578


namespace NUMINAMATH_CALUDE_sum_of_roots_l235_23521

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 5*x - 17 = 0)
  (hy : y^3 - 3*y^2 + 5*y + 11 = 0) : 
  x + y = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l235_23521


namespace NUMINAMATH_CALUDE_B_power_2017_l235_23502

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_2017 : B^2017 = B := by sorry

end NUMINAMATH_CALUDE_B_power_2017_l235_23502


namespace NUMINAMATH_CALUDE_proposition_logic_l235_23531

theorem proposition_logic (p q : Prop) 
  (h_p_false : ¬p) 
  (h_q_true : q) : 
  (¬(p ∧ q)) ∧ 
  (p ∨ q) ∧ 
  (¬p) ∧ 
  (¬(¬q)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_logic_l235_23531


namespace NUMINAMATH_CALUDE_exists_non_identity_same_image_l235_23559

/-- Given two finite groups G and H, and two surjective but non-injective homomorphisms φ and ψ from G to H,
    there exists a non-identity element g in G such that φ(g) = ψ(g). -/
theorem exists_non_identity_same_image 
  {G H : Type*} [Group G] [Group H] [Fintype G] [Fintype H]
  (φ ψ : G →* H) 
  (hφ_surj : Function.Surjective φ) (hψ_surj : Function.Surjective ψ)
  (hφ_non_inj : ¬Function.Injective φ) (hψ_non_inj : ¬Function.Injective ψ) :
  ∃ g : G, g ≠ 1 ∧ φ g = ψ g := by
  sorry

end NUMINAMATH_CALUDE_exists_non_identity_same_image_l235_23559


namespace NUMINAMATH_CALUDE_imaginary_part_of_fraction_l235_23523

noncomputable def i : ℂ := Complex.I

theorem imaginary_part_of_fraction (z : ℂ) : z = 2016 / (1 + i) → Complex.im z = -1008 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_fraction_l235_23523


namespace NUMINAMATH_CALUDE_total_trees_after_planting_l235_23584

def current_trees : ℕ := 33
def new_trees : ℕ := 44

theorem total_trees_after_planting :
  current_trees + new_trees = 77 := by sorry

end NUMINAMATH_CALUDE_total_trees_after_planting_l235_23584


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l235_23513

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 10}
def B : Set ℝ := {x | -x^2 + 2 ≤ 2}

-- Define the open interval (1, 2]
def openClosedInterval : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem intersection_equals_interval : A ∩ B = openClosedInterval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l235_23513


namespace NUMINAMATH_CALUDE_three_rug_overlap_l235_23530

theorem three_rug_overlap (A B C X Y Z : ℝ) 
  (h1 : A + B + C = 200) 
  (h2 : X + Y + Z = 140) 
  (h3 : Y = 22) 
  (h4 : X + 2*Y + 3*Z = A + B + C) : 
  Z = 19 := by
sorry

end NUMINAMATH_CALUDE_three_rug_overlap_l235_23530


namespace NUMINAMATH_CALUDE_max_value_of_expression_l235_23516

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5*x + 6*y < 90) :
  ∃ (M : ℝ), M = 900 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → 5*a + 6*b < 90 → a*b*(90 - 5*a - 6*b) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l235_23516


namespace NUMINAMATH_CALUDE_shop_markup_problem_l235_23581

/-- A shop owner purchases goods at a discount and wants to mark them up for profit. -/
theorem shop_markup_problem (L : ℝ) (C : ℝ) (M : ℝ) (S : ℝ) :
  C = 0.75 * L →           -- Cost price is 75% of list price
  S = 1.3 * C →            -- Selling price is 130% of cost price
  S = 0.75 * M →           -- Selling price is 75% of marked price
  M = 1.3 * L              -- Marked price is 130% of list price
:= by sorry

end NUMINAMATH_CALUDE_shop_markup_problem_l235_23581


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l235_23588

theorem quadratic_equations_common_root (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x - 12 = 0 ∧ 3*x^2 - 8*x - 3*k = 0) →
  k = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l235_23588


namespace NUMINAMATH_CALUDE_max_balloons_is_400_l235_23596

def small_bag_cost : ℕ := 4
def small_bag_balloons : ℕ := 50
def medium_bag_cost : ℕ := 6
def medium_bag_balloons : ℕ := 75
def large_bag_cost : ℕ := 12
def large_bag_balloons : ℕ := 200
def budget : ℕ := 24

def max_balloons (budget small_cost small_balloons medium_cost medium_balloons large_cost large_balloons : ℕ) : ℕ := 
  sorry

theorem max_balloons_is_400 : 
  max_balloons budget small_bag_cost small_bag_balloons medium_bag_cost medium_bag_balloons large_bag_cost large_bag_balloons = 400 :=
by sorry

end NUMINAMATH_CALUDE_max_balloons_is_400_l235_23596


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l235_23541

theorem sum_of_a_and_b (a b : ℝ) (ha : |a| = 4) (hb : |b| = 7) (hab : a < b) :
  a + b = 3 ∨ a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l235_23541


namespace NUMINAMATH_CALUDE_salary_change_percentage_l235_23562

theorem salary_change_percentage (original : ℝ) (h : original > 0) :
  let decreased := original * (1 - 0.5)
  let increased := decreased * (1 + 0.5)
  increased = original * 0.75 ∧ (original - increased) / original = 0.25 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l235_23562


namespace NUMINAMATH_CALUDE_cosine_triple_angle_identity_l235_23563

theorem cosine_triple_angle_identity (x : ℝ) : 
  4 * Real.cos x * Real.cos (x + π/3) * Real.cos (x - π/3) = Real.cos (3*x) := by
  sorry

end NUMINAMATH_CALUDE_cosine_triple_angle_identity_l235_23563


namespace NUMINAMATH_CALUDE_exists_equal_digit_sum_l235_23529

-- Define an arithmetic progression
def arithmeticProgression (a₀ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₀ + n * d

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_equal_digit_sum (a₀ : ℕ) (d : ℕ) (h : d ≠ 0) :
  ∃ (m n : ℕ), m ≠ n ∧ 
    sumOfDigits (arithmeticProgression a₀ d m) = sumOfDigits (arithmeticProgression a₀ d n) := by
  sorry


end NUMINAMATH_CALUDE_exists_equal_digit_sum_l235_23529


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l235_23560

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (α β : Plane) (m : Line) 
  (h1 : parallel_planes α β) 
  (h2 : contained_in m β) : 
  parallel_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l235_23560


namespace NUMINAMATH_CALUDE_max_posters_purchasable_l235_23556

def initial_amount : ℕ := 20
def book1_price : ℕ := 8
def book2_price : ℕ := 4
def poster_price : ℕ := 4

theorem max_posters_purchasable :
  (initial_amount - book1_price - book2_price) / poster_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_posters_purchasable_l235_23556


namespace NUMINAMATH_CALUDE_hockey_league_games_l235_23572

structure Team where
  games_played : ℕ
  games_won : ℕ
  win_ratio : ℚ

def team_X : Team → Prop
| t => t.win_ratio = 3/4

def team_Y : Team → Prop
| t => t.win_ratio = 2/3

theorem hockey_league_games (X Y : Team) : 
  team_X X → team_Y Y → 
  Y.games_played = X.games_played + 12 →
  Y.games_won = X.games_won + 4 →
  X.games_played = 48 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l235_23572


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l235_23526

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 9
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -3 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l235_23526


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l235_23575

/-- The total number of handshakes -/
def total_handshakes : ℕ := 325

/-- The number of gymnasts -/
def n : ℕ := 26

/-- The number of handshakes between gymnasts -/
def gymnast_handshakes : ℕ := n * (n - 1) / 2

/-- The number of handshakes by the first coach -/
def coach1_handshakes : ℕ := 0

/-- The number of handshakes by the second coach -/
def coach2_handshakes : ℕ := total_handshakes - gymnast_handshakes - coach1_handshakes

theorem min_coach_handshakes :
  gymnast_handshakes + coach1_handshakes + coach2_handshakes = total_handshakes ∧
  coach1_handshakes = 0 ∧
  coach2_handshakes ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l235_23575


namespace NUMINAMATH_CALUDE_complex_cube_eq_negative_eight_l235_23517

theorem complex_cube_eq_negative_eight :
  (1 + Complex.I * Real.sqrt 3) ^ 3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_eq_negative_eight_l235_23517


namespace NUMINAMATH_CALUDE_magnitude_of_z_l235_23548

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The given complex number -/
def z : ℂ := (1 - 2*i) * i

/-- Theorem stating that the magnitude of z is √5 -/
theorem magnitude_of_z : Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l235_23548


namespace NUMINAMATH_CALUDE_odd_plus_one_even_implies_f_four_zero_l235_23535

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_plus_one_even_implies_f_four_zero (f : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even (fun x ↦ f (x + 1))) : f 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_plus_one_even_implies_f_four_zero_l235_23535


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_7_range_of_m_for_solution_exists_l235_23511

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| + |2*x - 3|

-- Theorem for the solution set of f(x) > 7
theorem solution_set_f_greater_than_7 :
  {x : ℝ | f x > 7} = {x : ℝ | x < -3/2 ∨ x > 2} :=
sorry

-- Theorem for the range of m
theorem range_of_m_for_solution_exists :
  {m : ℝ | ∃ x, f x ≤ |3*m - 2|} = {m : ℝ | m ≤ -1 ∨ m ≥ 7/3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_7_range_of_m_for_solution_exists_l235_23511


namespace NUMINAMATH_CALUDE_tims_medical_cost_tims_out_of_pocket_cost_l235_23590

/-- Calculates the out-of-pocket cost for Tim's medical visit --/
theorem tims_medical_cost (mri_cost : ℚ) (doctor_rate : ℚ) (exam_time : ℚ) 
  (visit_fee : ℚ) (insurance_coverage : ℚ) : ℚ :=
  let total_cost := mri_cost + doctor_rate * exam_time / 2 + visit_fee
  let insurance_payment := total_cost * insurance_coverage
  total_cost - insurance_payment

/-- Proves that Tim's out-of-pocket cost is $300 --/
theorem tims_out_of_pocket_cost : 
  tims_medical_cost 1200 300 (1/2) 150 (4/5) = 300 := by
  sorry

end NUMINAMATH_CALUDE_tims_medical_cost_tims_out_of_pocket_cost_l235_23590


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l235_23550

theorem quadratic_root_implies_a (a : ℝ) : 
  let S := {x : ℝ | x^2 + 2*x + a = 0}
  (-1 : ℝ) ∈ S → a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l235_23550


namespace NUMINAMATH_CALUDE_school_play_tickets_l235_23557

theorem school_play_tickets (total_money : ℕ) (adult_price : ℕ) (child_price : ℕ) (total_tickets : ℕ) :
  total_money = 104 →
  adult_price = 6 →
  child_price = 4 →
  total_tickets = 21 →
  ∃ (adult_tickets : ℕ) (child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_money ∧
    child_tickets = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_school_play_tickets_l235_23557


namespace NUMINAMATH_CALUDE_lunch_group_size_l235_23500

/-- The number of people having lunch, including Benny -/
def num_people : ℕ := 3

/-- The cost of one lunch special in dollars -/
def lunch_cost : ℕ := 8

/-- The total bill in dollars -/
def total_bill : ℕ := 24

theorem lunch_group_size :
  num_people * lunch_cost = total_bill :=
by sorry

end NUMINAMATH_CALUDE_lunch_group_size_l235_23500


namespace NUMINAMATH_CALUDE_quadrilateral_theorem_l235_23583

-- Define a quadrilateral
structure Quadrilateral :=
(A B C D : ℝ × ℝ)

-- Define the angle between two vectors
def angle (v w : ℝ × ℝ) : ℝ := sorry

-- Define the length of a vector
def length (v : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem quadrilateral_theorem (q : Quadrilateral) 
  (h : angle (q.C.1 - q.A.1, q.C.2 - q.A.2) (q.A.1 - q.C.1, q.A.2 - q.C.2) = 120) :
  (length (q.A.1 - q.C.1, q.A.2 - q.C.2) * length (q.B.1 - q.D.1, q.B.2 - q.D.2))^2 =
  (length (q.A.1 - q.B.1, q.A.2 - q.B.2) * length (q.C.1 - q.D.1, q.C.2 - q.D.2))^2 +
  (length (q.B.1 - q.C.1, q.B.2 - q.C.2) * length (q.A.1 - q.D.1, q.A.2 - q.D.2))^2 +
  length (q.A.1 - q.B.1, q.A.2 - q.B.2) * length (q.B.1 - q.C.1, q.B.2 - q.C.2) *
  length (q.C.1 - q.D.1, q.C.2 - q.D.2) * length (q.D.1 - q.A.1, q.D.2 - q.A.2) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_theorem_l235_23583


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l235_23576

theorem complex_sum_theorem (A O P S : ℂ) 
  (hA : A = 2 + I) 
  (hO : O = 3 - 2*I) 
  (hP : P = 1 + I) 
  (hS : S = 4 + 3*I) : 
  A - O + P + S = 4 + 7*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l235_23576


namespace NUMINAMATH_CALUDE_big_n_conference_teams_l235_23508

/-- The number of games in a round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of teams in the BIG N conference -/
theorem big_n_conference_teams : ∃ (n : ℕ), n > 0 ∧ num_games n = 21 :=
  sorry

end NUMINAMATH_CALUDE_big_n_conference_teams_l235_23508


namespace NUMINAMATH_CALUDE_max_sum_semi_axes_l235_23505

/-- The maximum sum of semi-axes of an ellipse and hyperbola with the same foci -/
theorem max_sum_semi_axes (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, x^2/25 + y^2/m^2 = 1) →
  (∃ x y : ℝ, x^2/7 - y^2/n^2 = 1) →
  (25 - m^2 = 7 + n^2) →
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 
    (∃ x y : ℝ, x^2/25 + y^2/m'^2 = 1) →
    (∃ x y : ℝ, x^2/7 - y^2/n'^2 = 1) →
    (25 - m'^2 = 7 + n'^2) →
    m + n ≥ m' + n') →
  m + n = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_semi_axes_l235_23505


namespace NUMINAMATH_CALUDE_percentage_of_seniors_with_cars_l235_23567

theorem percentage_of_seniors_with_cars :
  ∀ (total_students : ℕ) 
    (seniors : ℕ) 
    (lower_grades : ℕ) 
    (lower_grades_car_percentage : ℚ) 
    (total_car_percentage : ℚ),
  total_students = 1200 →
  seniors = 300 →
  lower_grades = 900 →
  lower_grades_car_percentage = 1/10 →
  total_car_percentage = 1/5 →
  (↑seniors * (seniors_car_percentage : ℚ) + ↑lower_grades * lower_grades_car_percentage) / ↑total_students = total_car_percentage →
  seniors_car_percentage = 1/2 :=
by
  sorry

#check percentage_of_seniors_with_cars

end NUMINAMATH_CALUDE_percentage_of_seniors_with_cars_l235_23567


namespace NUMINAMATH_CALUDE_polynomial_factorization_and_trig_inequality_l235_23570

theorem polynomial_factorization_and_trig_inequality :
  (∀ x : ℂ, x^12 + x^9 + x^6 + x^3 + 1 = (x^4 + x^3 + x^2 + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1)) ∧
  (∀ θ : ℝ, 5 + 8 * Real.cos θ + 4 * Real.cos (2 * θ) + Real.cos (3 * θ) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_and_trig_inequality_l235_23570


namespace NUMINAMATH_CALUDE_march_greatest_drop_l235_23512

/-- Represents the months in the first half of 2021 -/
inductive Month
| january
| february
| march
| april
| may
| june

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.january  => -3.00
  | Month.february => 1.50
  | Month.march    => -4.50
  | Month.april    => 2.00
  | Month.may      => -1.00
  | Month.june     => 0.50

/-- The month with the greatest price drop -/
def greatest_drop : Month := Month.march

theorem march_greatest_drop :
  ∀ m : Month, price_change greatest_drop ≤ price_change m :=
by sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l235_23512


namespace NUMINAMATH_CALUDE_cuboid_volume_l235_23539

/-- The volume of a cuboid with edges 2 cm, 5 cm, and 3 cm is 30 cubic centimeters. -/
theorem cuboid_volume : 
  ∀ (length width height : ℝ), 
    length = 2 → width = 5 → height = 3 → 
    length * width * height = 30 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l235_23539


namespace NUMINAMATH_CALUDE_base_ten_to_seven_l235_23510

theorem base_ten_to_seven : 2023 = 5 * 7^3 + 6 * 7^2 + 2 * 7^1 + 0 * 7^0 := by
  sorry

end NUMINAMATH_CALUDE_base_ten_to_seven_l235_23510


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_15_l235_23565

-- Define a function to calculate the sum of digits of a natural number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the function g(n) as the sum of digits of 1/3^n to the right of the decimal point
def g (n : ℕ) : ℕ := sumOfDigits (10^n / 3^n)

-- Theorem statement
theorem smallest_n_exceeding_15 :
  (∀ k < 6, g k ≤ 15) ∧ g 6 > 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_15_l235_23565


namespace NUMINAMATH_CALUDE_disk_color_difference_l235_23552

/-- Given a bag of disks with a specific color ratio and total count, 
    calculate the difference between green and blue disks. -/
theorem disk_color_difference 
  (total_disks : ℕ) 
  (blue_ratio yellow_ratio green_ratio red_ratio : ℕ) 
  (h_total : total_disks = 132)
  (h_ratio : blue_ratio + yellow_ratio + green_ratio + red_ratio = 22)
  (h_blue : blue_ratio = 3)
  (h_yellow : yellow_ratio = 7)
  (h_green : green_ratio = 8)
  (h_red : red_ratio = 4) :
  green_ratio * (total_disks / (blue_ratio + yellow_ratio + green_ratio + red_ratio)) -
  blue_ratio * (total_disks / (blue_ratio + yellow_ratio + green_ratio + red_ratio)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_disk_color_difference_l235_23552


namespace NUMINAMATH_CALUDE_first_discount_percentage_l235_23561

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) : 
  original_price = 390 →
  final_price = 248.625 →
  ∃ (first_discount : ℝ),
    first_discount = 15 ∧
    final_price = original_price * (100 - first_discount) / 100 * 75 / 100 :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l235_23561


namespace NUMINAMATH_CALUDE_ohara_triple_x_value_l235_23546

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℝ) : Prop :=
  Real.sqrt (abs a) + Real.sqrt (abs b) = x

/-- Theorem: If (-49, 64, x) is an O'Hara triple, then x = 15 -/
theorem ohara_triple_x_value :
  ∀ x : ℝ, is_ohara_triple (-49) 64 x → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_x_value_l235_23546


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l235_23573

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 32)
  (h2 : failed_english = 56)
  (h3 : failed_both = 12) :
  100 - (failed_hindi + failed_english - failed_both) = 24 :=
by sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l235_23573


namespace NUMINAMATH_CALUDE_sin_two_theta_value_l235_23532

/-- If e^(2iθ) = (2 + i√5) / 3, then sin 2θ = √3 / 3 -/
theorem sin_two_theta_value (θ : ℝ) (h : Complex.exp (2 * θ * Complex.I) = (2 + Complex.I * Real.sqrt 5) / 3) : 
  Real.sin (2 * θ) = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_two_theta_value_l235_23532


namespace NUMINAMATH_CALUDE_point_Q_y_coordinate_product_l235_23538

theorem point_Q_y_coordinate_product : ∀ (y₁ y₂ : ℝ),
  (∃ (Q : ℝ × ℝ), 
    Q.1 = 4 ∧ 
    ((Q.1 - 1)^2 + (Q.2 - (-3))^2) = 10^2 ∧
    (Q.2 = y₁ ∨ Q.2 = y₂) ∧
    y₁ ≠ y₂) →
  y₁ * y₂ = -82 := by
sorry

end NUMINAMATH_CALUDE_point_Q_y_coordinate_product_l235_23538


namespace NUMINAMATH_CALUDE_x_squared_over_y_squared_equals_two_l235_23528

theorem x_squared_over_y_squared_equals_two
  (x y z : ℝ) 
  (x_pos : 0 < x) 
  (y_pos : 0 < y) 
  (z_pos : 0 < z) 
  (all_different : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h : y^2 / (x^2 - z^2) = (x^2 + y^2) / z^2)
  (h2 : (x^2 + y^2) / z^2 = x^2 / y^2) :
  x^2 / y^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_x_squared_over_y_squared_equals_two_l235_23528


namespace NUMINAMATH_CALUDE_square_difference_305_295_l235_23585

theorem square_difference_305_295 : (305 : ℤ)^2 - (295 : ℤ)^2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_305_295_l235_23585


namespace NUMINAMATH_CALUDE_flare_problem_l235_23582

-- Define the height function
def h (v : ℝ) (t : ℝ) : ℝ := v * t - 4.9 * t^2

-- State the theorem
theorem flare_problem (v : ℝ) :
  h v 5 = 245 →
  v = 73.5 ∧
  ∃ t1 t2 : ℝ, t1 = 5 ∧ t2 = 10 ∧ ∀ t, t1 < t ∧ t < t2 → h v t > 245 :=
by sorry

end NUMINAMATH_CALUDE_flare_problem_l235_23582


namespace NUMINAMATH_CALUDE_quadratic_reciprocity_legendre_symbol_two_l235_23501

-- Define the Legendre symbol
noncomputable def legendre_symbol (a p : ℕ) : ℤ := sorry

-- Quadratic reciprocity law
theorem quadratic_reciprocity (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hodd_p : Odd p) (hodd_q : Odd q) :
  legendre_symbol p q * legendre_symbol q p = (-1) ^ ((p - 1) * (q - 1) / 4) := by sorry

-- Legendre symbol of 2
theorem legendre_symbol_two (m : ℕ) (hm : Nat.Prime m) (hodd_m : Odd m) :
  legendre_symbol 2 m = (-1) ^ ((m^2 - 1) / 8) := by sorry

end NUMINAMATH_CALUDE_quadratic_reciprocity_legendre_symbol_two_l235_23501


namespace NUMINAMATH_CALUDE_cube_edge_length_l235_23537

/-- Given a cube with volume 3375 cm³, prove that the total length of its edges is 180 cm. -/
theorem cube_edge_length (V : ℝ) (h : V = 3375) : 
  12 * (V ^ (1/3 : ℝ)) = 180 :=
sorry

end NUMINAMATH_CALUDE_cube_edge_length_l235_23537


namespace NUMINAMATH_CALUDE_xiao_hong_age_expression_dad_age_when_xiao_hong_is_seven_l235_23564

-- Define Dad's age
def dad_age : ℕ → ℕ := λ a => a

-- Define Xiao Hong's age as a function of Dad's age
def xiao_hong_age : ℕ → ℚ := λ a => (a - 3) / 4

-- Theorem for Xiao Hong's age expression
theorem xiao_hong_age_expression (a : ℕ) :
  xiao_hong_age a = (a - 3) / 4 :=
sorry

-- Theorem for Dad's age when Xiao Hong is 7
theorem dad_age_when_xiao_hong_is_seven :
  ∃ a : ℕ, xiao_hong_age a = 7 ∧ dad_age a = 31 :=
sorry

end NUMINAMATH_CALUDE_xiao_hong_age_expression_dad_age_when_xiao_hong_is_seven_l235_23564


namespace NUMINAMATH_CALUDE_intersection_M_N_l235_23533

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l235_23533


namespace NUMINAMATH_CALUDE_geometric_series_sum_l235_23514

/-- Sum of a finite geometric series -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 6 terms of the geometric series with first term 1/4 and common ratio 1/4 -/
theorem geometric_series_sum :
  geometricSum (1/4 : ℚ) (1/4 : ℚ) 6 = 4095/12288 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l235_23514


namespace NUMINAMATH_CALUDE_deck_size_proof_l235_23569

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b : ℚ) = 1/3 → 
  (r : ℚ) / (r + b + 4 : ℚ) = 1/4 → 
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_proof_l235_23569


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l235_23547

-- Define propositions p and q
def p (x : ℝ) : Prop := abs x < 1
def q (x : ℝ) : Prop := x^2 + x - 6 < 0

-- Theorem statement
theorem not_p_necessary_not_sufficient_for_not_q :
  (∀ x, ¬(q x) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ q x) :=
sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l235_23547


namespace NUMINAMATH_CALUDE_function_period_l235_23598

/-- Given a constant a and a function f: ℝ → ℝ that satisfies
    f(x) = (f(x-a) - 1) / (f(x-a) + 1) for all x ∈ ℝ,
    prove that f has period 4a. -/
theorem function_period (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x, f x = (f (x - a) - 1) / (f (x - a) + 1)) :
  ∀ x, f (x + 4*a) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_period_l235_23598


namespace NUMINAMATH_CALUDE_red_surface_fraction_is_one_l235_23586

/-- Represents a cube with its edge length and number of smaller cubes -/
structure Cube where
  edge_length : ℕ
  num_small_cubes : ℕ

/-- Represents the composition of the cube in terms of colored smaller cubes -/
structure CubeComposition where
  total_cubes : Cube
  red_cubes : ℕ
  blue_cubes : ℕ

/-- The fraction of the surface area of the larger cube that is red -/
def red_surface_fraction (c : CubeComposition) : ℚ :=
  sorry

/-- The theorem stating the fraction of red surface area -/
theorem red_surface_fraction_is_one (c : CubeComposition) 
  (h1 : c.total_cubes.edge_length = 4)
  (h2 : c.total_cubes.num_small_cubes = 64)
  (h3 : c.red_cubes = 40)
  (h4 : c.blue_cubes = 24)
  (h5 : c.red_cubes + c.blue_cubes = c.total_cubes.num_small_cubes) :
  red_surface_fraction c = 1 := by
  sorry

end NUMINAMATH_CALUDE_red_surface_fraction_is_one_l235_23586


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l235_23555

theorem factor_difference_of_squares (x : ℝ) : 4 * x^2 - 144 = 4 * (x - 6) * (x + 6) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l235_23555


namespace NUMINAMATH_CALUDE_division_problem_l235_23522

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 217 →
  divisor = 4 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 54 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l235_23522


namespace NUMINAMATH_CALUDE_fruits_in_box_l235_23504

/-- The number of fruits in a box after adding persimmons and apples -/
theorem fruits_in_box (persimmons apples : ℕ) : persimmons = 2 → apples = 7 → persimmons + apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_fruits_in_box_l235_23504


namespace NUMINAMATH_CALUDE_chromatic_number_lower_bound_l235_23592

/-- A simple graph -/
structure Graph (V : Type*) where
  adj : V → V → Prop

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- The maximum size of cliques in a graph -/
def omega (G : Graph V) : ℕ :=
  sorry

/-- The maximum size of independent sets in a graph -/
def omegaBar (G : Graph V) : ℕ :=
  sorry

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph V) : ℕ :=
  sorry

/-- The main theorem -/
theorem chromatic_number_lower_bound (G : Graph V) :
  chromaticNumber G ≥ max (omega G) (Fintype.card V / omegaBar G) :=
sorry

end NUMINAMATH_CALUDE_chromatic_number_lower_bound_l235_23592


namespace NUMINAMATH_CALUDE_no_factors_l235_23518

/-- The main polynomial -/
def p (x : ℝ) : ℝ := x^4 - 4*x^2 + 16

/-- Potential factor 1 -/
def f1 (x : ℝ) : ℝ := x^2 + 4

/-- Potential factor 2 -/
def f2 (x : ℝ) : ℝ := x - 2

/-- Potential factor 3 -/
def f3 (x : ℝ) : ℝ := x^2 - 4

/-- Potential factor 4 -/
def f4 (x : ℝ) : ℝ := x^2 + 2*x + 4

/-- Theorem stating that none of the given polynomials are factors of p -/
theorem no_factors : 
  (∀ x, p x ≠ 0 → f1 x ≠ 0) ∧ 
  (∀ x, p x ≠ 0 → f2 x ≠ 0) ∧ 
  (∀ x, p x ≠ 0 → f3 x ≠ 0) ∧ 
  (∀ x, p x ≠ 0 → f4 x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_factors_l235_23518


namespace NUMINAMATH_CALUDE_function_value_at_symmetric_point_l235_23594

/-- Given a function f(x) = a * sin³(x) + b * tan(x) + 1 where f(2) = 3,
    prove that f(2π - 2) = -1 -/
theorem function_value_at_symmetric_point
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * (Real.sin x)^3 + b * Real.tan x + 1)
  (h2 : f 2 = 3) :
  f (2 * Real.pi - 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_symmetric_point_l235_23594


namespace NUMINAMATH_CALUDE_job_completion_time_l235_23524

theorem job_completion_time (x : ℝ) : 
  x > 0 → -- A's completion time is positive
  4 * (1/x + 1/20) = 1 - 0.5333333333333333 → -- Condition from working together
  x = 15 := by
    sorry

end NUMINAMATH_CALUDE_job_completion_time_l235_23524


namespace NUMINAMATH_CALUDE_division_problem_l235_23599

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by sorry

end NUMINAMATH_CALUDE_division_problem_l235_23599


namespace NUMINAMATH_CALUDE_equation_transformation_l235_23506

theorem equation_transformation (a b : ℝ) : 
  (∀ x, x^2 - 6*x - 5 = 0 ↔ (x + a)^2 = b) → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l235_23506


namespace NUMINAMATH_CALUDE_school_picnic_attendees_l235_23579

/-- The number of attendees at the school picnic. -/
def num_attendees : ℕ := 1006

/-- The total number of plates prepared by the school. -/
def total_plates : ℕ := 2015 - num_attendees

theorem school_picnic_attendees :
  (∀ n : ℕ, n ≤ num_attendees → total_plates - (n - 1) > 0) ∧
  (total_plates - (num_attendees - 1) = 4) ∧
  (num_attendees + total_plates = 2015) :=
sorry

end NUMINAMATH_CALUDE_school_picnic_attendees_l235_23579


namespace NUMINAMATH_CALUDE_parabola_directrix_l235_23540

/-- The equation of the directrix of the parabola y = x^2 is y = -1/4 -/
theorem parabola_directrix : ∃ (k : ℝ), k = -1/4 ∧
  ∀ (x y : ℝ), y = x^2 → (x = 0 ∨ (x^2 + (y - k)^2) / (2 * (y - k)) = k) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l235_23540


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l235_23549

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (ratio : ℝ),
    r = 7 →
    ratio = 3 →
    let d := 2 * r
    let w := d
    let l := ratio * w
    l * w = 588 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l235_23549


namespace NUMINAMATH_CALUDE_fraction_calculation_l235_23519

theorem fraction_calculation : (3 / 4 + 2 + 1 / 3) / (1 + 1 / 2) = 37 / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l235_23519


namespace NUMINAMATH_CALUDE_seven_eighths_of_64_l235_23507

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_64_l235_23507


namespace NUMINAMATH_CALUDE_apples_on_table_l235_23554

/-- The number of green apples on the table -/
def green_apples : ℕ := 2

/-- The number of red apples on the table -/
def red_apples : ℕ := 3

/-- The number of yellow apples on the table -/
def yellow_apples : ℕ := 14

/-- The total number of apples on the table -/
def total_apples : ℕ := green_apples + red_apples + yellow_apples

theorem apples_on_table : total_apples = 19 := by
  sorry

end NUMINAMATH_CALUDE_apples_on_table_l235_23554


namespace NUMINAMATH_CALUDE_det_B_equals_five_l235_23543

theorem det_B_equals_five (b c : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![b, 3; -1, c]
  B + 3 * B⁻¹ = 0 → Matrix.det B = 5 := by
sorry

end NUMINAMATH_CALUDE_det_B_equals_five_l235_23543


namespace NUMINAMATH_CALUDE_xy_ratio_values_l235_23503

theorem xy_ratio_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : 2 * x^2 + 2 * y^2 = 5 * x * y) : 
  (x + y) / (x - y) = 3 ∨ (x + y) / (x - y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_xy_ratio_values_l235_23503


namespace NUMINAMATH_CALUDE_average_tip_fraction_l235_23551

-- Define the weekly tip fractions
def week1_tip_fraction : ℚ := 2/4
def week2_tip_fraction : ℚ := 3/8
def week3_tip_fraction : ℚ := 5/16
def week4_tip_fraction : ℚ := 1/4

-- Define the number of weeks
def num_weeks : ℕ := 4

-- Theorem statement
theorem average_tip_fraction :
  (week1_tip_fraction + week2_tip_fraction + week3_tip_fraction + week4_tip_fraction) / num_weeks = 23/64 := by
  sorry

end NUMINAMATH_CALUDE_average_tip_fraction_l235_23551


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l235_23589

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the internals of the plane for this problem
  dummy : Unit

/-- A line in 3D space -/
structure Line3D where
  -- We don't need to define the internals of the line for this problem
  dummy : Unit

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def perpendicular_planes (p1 : Plane3D) (p2 : Plane3D) : Prop :=
  sorry

/-- The main theorem -/
theorem line_perp_parallel_implies_planes_perp 
  (a b g : Plane3D) (l : Line3D) 
  (h1 : a ≠ b) (h2 : a ≠ g) (h3 : b ≠ g)
  (h4 : perpendicular l a) (h5 : parallel l b) : 
  perpendicular_planes a b :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l235_23589


namespace NUMINAMATH_CALUDE_ball_selection_problem_l235_23520

/-- The number of red balls in the box -/
def red_balls : ℕ := 12

/-- The number of blue balls in the box -/
def blue_balls : ℕ := 7

/-- The total number of balls in the box -/
def total_balls : ℕ := red_balls + blue_balls

/-- The number of ways to select 3 red balls and 2 blue balls -/
def ways_to_select : ℕ := Nat.choose red_balls 3 * Nat.choose blue_balls 2

/-- The probability of drawing 2 blue balls first, then 1 red ball -/
def prob_draw : ℚ :=
  (Nat.choose blue_balls 2 * Nat.choose red_balls 1) / Nat.choose total_balls 3

/-- The final result -/
theorem ball_selection_problem :
  ways_to_select * prob_draw = 388680 / 323 := by sorry

end NUMINAMATH_CALUDE_ball_selection_problem_l235_23520


namespace NUMINAMATH_CALUDE_divisor_properties_l235_23580

def N (a b c : ℕ) (α β γ : ℕ) : ℕ := a^α * b^β * c^γ

variable (a b c α β γ : ℕ)
variable (ha : Nat.Prime a)
variable (hb : Nat.Prime b)
variable (hc : Nat.Prime c)

theorem divisor_properties :
  let n := N a b c α β γ
  -- Total number of divisors
  ∃ d : ℕ → ℕ, d n = (α + 1) * (β + 1) * (γ + 1) ∧
  -- Product of equidistant divisors
  ∀ x y : ℕ, x ∣ n → y ∣ n → x * y = n →
    ∃ z : ℕ, z ∣ n ∧ z * z = n ∧
  -- Product of all divisors
  ∃ P : ℕ, P = n ^ ((α + 1) * (β + 1) * (γ + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_divisor_properties_l235_23580


namespace NUMINAMATH_CALUDE_equation_solution_l235_23591

theorem equation_solution : ∃! x : ℝ, (2 / (x - 3) = 3 / (x - 6)) ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l235_23591


namespace NUMINAMATH_CALUDE_phi_value_l235_23525

theorem phi_value (φ : Real) (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : Real.sqrt 2 * Real.cos (20 * Real.pi / 180) = Real.sin φ - Real.cos φ) : 
  φ = 25 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_phi_value_l235_23525


namespace NUMINAMATH_CALUDE_second_wood_weight_l235_23574

/-- Represents a square piece of wood -/
structure Wood where
  side : ℝ
  weight : ℝ

/-- The weight of a square piece of wood is proportional to its area -/
axiom weight_prop_area {w1 w2 : Wood} :
  w1.weight / w2.weight = (w1.side ^ 2) / (w2.side ^ 2)

/-- Given two pieces of wood with specific properties, prove the weight of the second piece -/
theorem second_wood_weight (w1 w2 : Wood)
  (h1 : w1.side = 4)
  (h2 : w1.weight = 16)
  (h3 : w2.side = 6) :
  w2.weight = 36 := by
  sorry

#check second_wood_weight

end NUMINAMATH_CALUDE_second_wood_weight_l235_23574


namespace NUMINAMATH_CALUDE_inequalities_proof_l235_23509

theorem inequalities_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  2/(a-1) + 1/(b-2) ≥ 2 ∧ 2*a + b ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l235_23509


namespace NUMINAMATH_CALUDE_periodic_function_proof_l235_23515

open Real

theorem periodic_function_proof (a : ℚ) (b d c : ℝ) 
  (f : ℝ → ℝ) 
  (h_range : ∀ x, f x ∈ Set.Icc (-1) 1)
  (h_eq : ∀ x, f (x + a + b) - f (x + b) = c * (x + 2 * a + ⌊x⌋ - 2 * ⌊x + a⌋ - ⌊b⌋) + d) :
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end NUMINAMATH_CALUDE_periodic_function_proof_l235_23515
