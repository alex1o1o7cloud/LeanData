import Mathlib

namespace NUMINAMATH_CALUDE_certain_number_equation_l2144_214439

theorem certain_number_equation (x : ℝ) : 0.85 * 40 = (4/5) * x + 14 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2144_214439


namespace NUMINAMATH_CALUDE_prob_all_boys_prob_two_boys_one_girl_l2144_214444

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def select_num : ℕ := 3

/-- The probability of selecting 3 boys out of the total 6 people -/
theorem prob_all_boys : 
  (Nat.choose num_boys select_num : ℚ) / (Nat.choose total_people select_num) = 1 / 5 := by
  sorry

/-- The probability of selecting 2 boys and 1 girl out of the total 6 people -/
theorem prob_two_boys_one_girl : 
  ((Nat.choose num_boys 2 * Nat.choose num_girls 1) : ℚ) / (Nat.choose total_people select_num) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_boys_prob_two_boys_one_girl_l2144_214444


namespace NUMINAMATH_CALUDE_range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l2144_214405

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

-- Define proposition q
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Theorem for question 1
theorem range_of_m_when_p_true : 
  ∀ m : ℝ, p m ↔ m > 2 := sorry

-- Theorem for question 2
theorem range_of_m_when_p_and_q_false_p_or_q_true :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) := sorry

end NUMINAMATH_CALUDE_range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l2144_214405


namespace NUMINAMATH_CALUDE_inequality_problem_l2144_214449

theorem inequality_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l2144_214449


namespace NUMINAMATH_CALUDE_function_value_at_shifted_point_l2144_214435

/-- Given a function f(x) = a * tan³(x) + b * sin(x) + 1 where f(4) = 5, prove that f(2π - 4) = -3 -/
theorem function_value_at_shifted_point 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.tan x ^ 3 + b * Real.sin x + 1) 
  (h2 : f 4 = 5) : 
  f (2 * Real.pi - 4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_shifted_point_l2144_214435


namespace NUMINAMATH_CALUDE_max_elves_theorem_l2144_214429

/-- Represents the type of inhabitant -/
inductive InhabitantType
| Elf
| Dwarf

/-- Represents whether an inhabitant wears a cap -/
inductive CapStatus
| WithCap
| WithoutCap

/-- Represents the statement an inhabitant can make -/
inductive Statement
| RightIsElf
| RightHasCap

/-- Represents an inhabitant in the line -/
structure Inhabitant :=
  (type : InhabitantType)
  (capStatus : CapStatus)
  (statement : Statement)

/-- Determines if an inhabitant tells the truth based on their type and cap status -/
def tellsTruth (i : Inhabitant) : Prop :=
  match i.type, i.capStatus with
  | InhabitantType.Elf, CapStatus.WithoutCap => True
  | InhabitantType.Elf, CapStatus.WithCap => False
  | InhabitantType.Dwarf, CapStatus.WithoutCap => False
  | InhabitantType.Dwarf, CapStatus.WithCap => True

/-- Represents the line of inhabitants -/
def Line := Vector Inhabitant 60

/-- Checks if the line configuration is valid according to the problem rules -/
def isValidLine (line : Line) : Prop := sorry

/-- Counts the number of elves without caps in the line -/
def countElvesWithoutCaps (line : Line) : Nat := sorry

/-- Counts the number of elves with caps in the line -/
def countElvesWithCaps (line : Line) : Nat := sorry

/-- Main theorem: Maximum number of elves without caps is 59 and with caps is 30 -/
theorem max_elves_theorem (line : Line) (h : isValidLine line) : 
  countElvesWithoutCaps line ≤ 59 ∧ countElvesWithCaps line ≤ 30 := by sorry

end NUMINAMATH_CALUDE_max_elves_theorem_l2144_214429


namespace NUMINAMATH_CALUDE_juan_cars_count_l2144_214406

theorem juan_cars_count (num_bicycles num_pickup_trucks num_tricycles total_tires : ℕ)
  (h1 : num_bicycles = 3)
  (h2 : num_pickup_trucks = 8)
  (h3 : num_tricycles = 1)
  (h4 : total_tires = 101)
  (h5 : ∀ (num_cars : ℕ), total_tires = 4 * num_cars + 2 * num_bicycles + 4 * num_pickup_trucks + 3 * num_tricycles) :
  ∃ (num_cars : ℕ), num_cars = 15 ∧ total_tires = 4 * num_cars + 2 * num_bicycles + 4 * num_pickup_trucks + 3 * num_tricycles :=
by
  sorry

end NUMINAMATH_CALUDE_juan_cars_count_l2144_214406


namespace NUMINAMATH_CALUDE_first_episode_length_l2144_214455

/-- Given a series with four episodes, where the second episode is 62 minutes long,
    the third episode is 65 minutes long, the fourth episode is 55 minutes long,
    and the total duration of all four episodes is 4 hours,
    prove that the first episode is 58 minutes long. -/
theorem first_episode_length :
  ∀ (episode1 episode2 episode3 episode4 : ℕ),
  episode2 = 62 →
  episode3 = 65 →
  episode4 = 55 →
  episode1 + episode2 + episode3 + episode4 = 4 * 60 →
  episode1 = 58 :=
by
  sorry


end NUMINAMATH_CALUDE_first_episode_length_l2144_214455


namespace NUMINAMATH_CALUDE_games_this_month_l2144_214446

/-- Represents the number of football games Nancy attended or plans to attend -/
structure FootballGames where
  total : Nat
  lastMonth : Nat
  nextMonth : Nat

/-- Theorem stating that Nancy attended 9 games this month -/
theorem games_this_month (nancy : FootballGames) 
  (h1 : nancy.total = 24) 
  (h2 : nancy.lastMonth = 8) 
  (h3 : nancy.nextMonth = 7) : 
  nancy.total - nancy.lastMonth - nancy.nextMonth = 9 := by
  sorry

#check games_this_month

end NUMINAMATH_CALUDE_games_this_month_l2144_214446


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2144_214464

theorem quadratic_factorization (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 15 = (m * x + n) * (p * x + q)) →
  (∃ k : ℤ, b = 2 * k) ∧ 
  ¬(∀ k : ℤ, ∃ (m n p q : ℤ), 15 * x^2 + (2 * k) * x + 15 = (m * x + n) * (p * x + q)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2144_214464


namespace NUMINAMATH_CALUDE_louise_cakes_proof_l2144_214423

/-- The number of cakes Louise needs for the gathering -/
def total_cakes : ℕ := 60

/-- The number of cakes Louise has already baked -/
def baked_cakes : ℕ := total_cakes / 2

/-- The number of cakes Louise bakes on the second day -/
def second_day_bakes : ℕ := (total_cakes - baked_cakes) / 2

/-- The number of cakes Louise bakes on the third day -/
def third_day_bakes : ℕ := (total_cakes - baked_cakes - second_day_bakes) / 3

/-- The number of cakes left to bake after the third day -/
def remaining_cakes : ℕ := total_cakes - baked_cakes - second_day_bakes - third_day_bakes

theorem louise_cakes_proof : remaining_cakes = 10 := by
  sorry

#eval total_cakes
#eval remaining_cakes

end NUMINAMATH_CALUDE_louise_cakes_proof_l2144_214423


namespace NUMINAMATH_CALUDE_root_not_sufficient_for_bisection_l2144_214437

-- Define a continuous function on a closed interval
def ContinuousOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  Continuous f ∧ a ≤ b

-- Define the condition for a function to have a root
def HasRoot (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

-- Define the conditions for the bisection method to be applicable
def BisectionApplicable (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOnInterval f a b ∧ f a * f b < 0

-- Theorem statement
theorem root_not_sufficient_for_bisection :
  ∃ f : ℝ → ℝ, HasRoot f ∧ ¬(∃ a b, BisectionApplicable f a b) :=
sorry

end NUMINAMATH_CALUDE_root_not_sufficient_for_bisection_l2144_214437


namespace NUMINAMATH_CALUDE_workshop_technicians_salary_l2144_214436

/-- Represents the average salary of technicians in a workshop -/
def average_salary_technicians (total_workers : ℕ) (technicians : ℕ) (avg_salary_all : ℚ) (avg_salary_others : ℚ) : ℚ :=
  let other_workers := total_workers - technicians
  let total_salary := (avg_salary_all * total_workers : ℚ)
  let other_salary := (avg_salary_others * other_workers : ℚ)
  let technicians_salary := total_salary - other_salary
  technicians_salary / technicians

/-- Theorem stating that the average salary of technicians is 1000 given the workshop conditions -/
theorem workshop_technicians_salary :
  average_salary_technicians 22 7 850 780 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_technicians_salary_l2144_214436


namespace NUMINAMATH_CALUDE_vector_dot_product_l2144_214498

theorem vector_dot_product (a b : ℝ × ℝ) : 
  (Real.sqrt 2 : ℝ) = Real.sqrt (a.1 ^ 2 + a.2 ^ 2) →
  2 = Real.sqrt (b.1 ^ 2 + b.2 ^ 2) →
  (3 * Real.pi / 4 : ℝ) = Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) * Real.sqrt (b.1 ^ 2 + b.2 ^ 2))) →
  (a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) : ℝ) = 6 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_l2144_214498


namespace NUMINAMATH_CALUDE_slope_product_implies_trajectory_l2144_214448

/-- The trajectory of point P(x,y) given fixed points A(-4,0) and B(4,0) -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 64 + y^2 / (64/9) = 1 ∧ x ≠ 4 ∧ x ≠ -4

/-- The slope product condition for point P(x,y) given fixed points A(-4,0) and B(4,0) -/
def slope_product_condition (x y : ℝ) : Prop :=
  (y / (x + 4)) * (y / (x - 4)) = -4/9 ∧ x ≠ 4 ∧ x ≠ -4

/-- Theorem stating that if the slope product condition is satisfied, 
    then the point lies on the trajectory -/
theorem slope_product_implies_trajectory (x y : ℝ) :
  slope_product_condition x y → trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_slope_product_implies_trajectory_l2144_214448


namespace NUMINAMATH_CALUDE_quotient_remainder_difference_l2144_214477

theorem quotient_remainder_difference (N : ℕ) : 
  N ≥ 75 → 
  N % 5 = 0 → 
  (∀ m : ℕ, m ≥ 75 ∧ m % 5 = 0 → m ≥ N) →
  (N / 5) - (N % 34) = 8 := by
  sorry

end NUMINAMATH_CALUDE_quotient_remainder_difference_l2144_214477


namespace NUMINAMATH_CALUDE_quadratic_sum_l2144_214466

-- Define the quadratic function
def f (x : ℝ) : ℝ := -4 * x^2 + 20 * x - 88

-- Define the general form a(x+b)^2 + c
def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum (a b c : ℝ) :
  (∀ x, f x = g a b c x) → a + b + c = -70.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2144_214466


namespace NUMINAMATH_CALUDE_not_A_union_B_equiv_l2144_214491

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) ≥ 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem not_A_union_B_equiv : (Aᶜ ∪ B) = {x : ℝ | x > -2} := by sorry

end NUMINAMATH_CALUDE_not_A_union_B_equiv_l2144_214491


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2144_214426

def U : Set Nat := {0, 1, 2, 3}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {0, 2, 3}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2144_214426


namespace NUMINAMATH_CALUDE_tangent_points_satisfy_locus_l2144_214427

/-- A conic section with focus at the origin and directrix x - d = 0 -/
structure ConicSection (d : ℝ) where
  -- Point on the conic section
  x : ℝ
  y : ℝ
  -- Eccentricity
  e : ℝ
  -- Conic section equation
  eq : x^2 + y^2 = e^2 * (x - d)^2

/-- A point of tangency on the conic section -/
structure TangentPoint (d : ℝ) extends ConicSection d where
  -- Tangent line has slope 1 (parallel to y = x)
  tangent_slope : (1 - e^2) * x + y + e^2 * d = 0

/-- The locus of points of tangency -/
def locus_equation (d : ℝ) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  y^2 - x*y + d*(x + y) = 0

/-- The main theorem: points of tangency satisfy the locus equation -/
theorem tangent_points_satisfy_locus (d : ℝ) (p : TangentPoint d) :
  locus_equation d (p.x, p.y) := by
  sorry


end NUMINAMATH_CALUDE_tangent_points_satisfy_locus_l2144_214427


namespace NUMINAMATH_CALUDE_xyz_sum_l2144_214494

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x.val * y.val + z.val = 47)
  (h2 : y.val * z.val + x.val = 47)
  (h3 : x.val * z.val + y.val = 47) :
  x.val + y.val + z.val = 48 :=
by sorry

end NUMINAMATH_CALUDE_xyz_sum_l2144_214494


namespace NUMINAMATH_CALUDE_investmentPlansCount_l2144_214460

/-- The number of ways to distribute 3 distinct projects among 6 locations,
    with no more than 2 projects per location. -/
def investmentPlans : ℕ :=
  Nat.descFactorial 6 3 + (Nat.choose 3 2 * Nat.descFactorial 6 2)

/-- Theorem stating that the number of distinct investment plans is 210. -/
theorem investmentPlansCount : investmentPlans = 210 := by
  sorry

end NUMINAMATH_CALUDE_investmentPlansCount_l2144_214460


namespace NUMINAMATH_CALUDE_man_rowing_speed_l2144_214425

/-- Proves that given a man's speed in still water and downstream speed, his upstream speed can be calculated. -/
theorem man_rowing_speed (v_still : ℝ) (v_downstream : ℝ) (h1 : v_still = 50) (h2 : v_downstream = 80) :
  v_still - (v_downstream - v_still) = 20 := by
  sorry

#check man_rowing_speed

end NUMINAMATH_CALUDE_man_rowing_speed_l2144_214425


namespace NUMINAMATH_CALUDE_range_of_a_l2144_214414

open Set Real

theorem range_of_a (p q : Prop) (h : ¬(p ∧ q)) : 
  ∀ a : ℝ, (∀ x ∈ Icc 0 1, a ≥ exp x) = p → 
  (∃ x₀ : ℝ, x₀^2 + 4*x₀ + a = 0) = q → 
  a ∈ Ioi 4 ∪ Iic (exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2144_214414


namespace NUMINAMATH_CALUDE_remainder_equality_l2144_214492

theorem remainder_equality (a b d s t : ℕ) 
  (h1 : a > b) 
  (h2 : a % d = s % d) 
  (h3 : b % d = t % d) : 
  ((a + 1) * (b + 1)) % d = ((s + 1) * (t + 1)) % d := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l2144_214492


namespace NUMINAMATH_CALUDE_shoe_size_increase_l2144_214484

/-- Represents the increase in length (in inches) for each unit increase in shoe size -/
def length_increase : ℝ := 0.2

/-- The smallest shoe size -/
def min_size : ℕ := 8

/-- The largest shoe size -/
def max_size : ℕ := 17

/-- The length of the size 15 shoe (in inches) -/
def size_15_length : ℝ := 10.4

/-- The ratio of the largest size length to the smallest size length -/
def size_ratio : ℝ := 1.2

theorem shoe_size_increase :
  (min_size : ℝ) + (max_size - min_size) * length_increase = (min_size : ℝ) * size_ratio ∧
  (min_size : ℝ) + (15 - min_size) * length_increase = size_15_length ∧
  length_increase = 0.2 := by sorry

end NUMINAMATH_CALUDE_shoe_size_increase_l2144_214484


namespace NUMINAMATH_CALUDE_trig_sum_problem_l2144_214486

theorem trig_sum_problem (α : Real) (h1 : 0 < α) (h2 : α < Real.pi) 
  (h3 : Real.sin α * Real.cos α = -1/2) : 
  1/(1 + Real.sin α) + 1/(1 + Real.cos α) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_problem_l2144_214486


namespace NUMINAMATH_CALUDE_sum_c_d_eq_eight_l2144_214465

/-- Two lines intersecting at a point -/
structure IntersectingLines where
  c : ℝ
  d : ℝ
  h : (2 * 4 + c = 16) ∧ (4 * 4 + d = 16)

/-- The sum of c and d for intersecting lines -/
def sum_c_d (lines : IntersectingLines) : ℝ := lines.c + lines.d

/-- Theorem: The sum of c and d equals 8 -/
theorem sum_c_d_eq_eight (lines : IntersectingLines) : sum_c_d lines = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_c_d_eq_eight_l2144_214465


namespace NUMINAMATH_CALUDE_count_numbers_with_6_or_7_proof_l2144_214418

/-- The number of integers from 1 to 512 (inclusive) in base 8 that contain at least one digit 6 or 7 -/
def count_numbers_with_6_or_7 : ℕ := 296

/-- The total number of integers we're considering -/
def total_numbers : ℕ := 512

/-- The base we're working in -/
def base : ℕ := 8

/-- The number of digits available in the restricted set (0-5) -/
def restricted_digits : ℕ := 6

/-- The number of digits needed to represent the largest number in our set in base 8 -/
def num_digits : ℕ := 3

theorem count_numbers_with_6_or_7_proof :
  count_numbers_with_6_or_7 = total_numbers - restricted_digits ^ num_digits :=
by sorry

end NUMINAMATH_CALUDE_count_numbers_with_6_or_7_proof_l2144_214418


namespace NUMINAMATH_CALUDE_two_draw_probability_l2144_214404

def red_chips : ℕ := 6
def blue_chips : ℕ := 4
def green_chips : ℕ := 2

def total_chips : ℕ := red_chips + blue_chips + green_chips

def prob_red_blue : ℚ := (red_chips * blue_chips + blue_chips * red_chips) / (total_chips * total_chips)
def prob_blue_green : ℚ := (blue_chips * green_chips + green_chips * blue_chips) / (total_chips * total_chips)

theorem two_draw_probability :
  prob_red_blue + prob_blue_green = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_two_draw_probability_l2144_214404


namespace NUMINAMATH_CALUDE_fraction_equality_l2144_214482

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / (a + 27 : ℚ) = 865 / 1000 → a = 173 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2144_214482


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_problem_l2144_214478

theorem gcd_lcm_sum_problem : Nat.gcd 40 60 + 2 * Nat.lcm 20 15 = 140 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_problem_l2144_214478


namespace NUMINAMATH_CALUDE_right_triangle_sin_complement_l2144_214473

theorem right_triangle_sin_complement (A B C : ℝ) :
  0 < A ∧ A < π / 2 →
  B = π / 2 →
  Real.sin A = 3 / 5 →
  Real.sin C = 4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sin_complement_l2144_214473


namespace NUMINAMATH_CALUDE_f_increasing_min_value_sum_tangent_line_l2144_214420

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 + 1/2

-- Statement 1: f(x) is monotonically increasing on [5π/6, π]
theorem f_increasing : ∀ x y, 5*Real.pi/6 ≤ x ∧ x < y ∧ y ≤ Real.pi → f x < f y := by sorry

-- Statement 2: The minimum value of f(x) + f(x + π/4) is -√2
theorem min_value_sum : ∃ m : ℝ, (∀ x, m ≤ f x + f (x + Real.pi/4)) ∧ m = -Real.sqrt 2 := by sorry

-- Statement 3: The line y = √3x - 1/2 is a tangent line to y = f(x)
theorem tangent_line : ∃ x₀ : ℝ, f x₀ = Real.sqrt 3 * x₀ - 1/2 ∧ 
  (∀ x, f x ≤ Real.sqrt 3 * x - 1/2) := by sorry

end NUMINAMATH_CALUDE_f_increasing_min_value_sum_tangent_line_l2144_214420


namespace NUMINAMATH_CALUDE_field_length_proof_l2144_214470

theorem field_length_proof (width : ℝ) (length : ℝ) (pond_side : ℝ) :
  length = 2 * width →
  pond_side = 9 →
  pond_side^2 = (1/8) * (length * width) →
  length = 36 := by
sorry

end NUMINAMATH_CALUDE_field_length_proof_l2144_214470


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_c_value_l2144_214459

theorem polynomial_equality_implies_c_value (a c : ℚ) 
  (h : ∀ x : ℚ, (x + 3) * (x + a) = x^2 + c*x + 8) : 
  c = 17/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_c_value_l2144_214459


namespace NUMINAMATH_CALUDE_quadratic_sum_l2144_214409

/-- Given a quadratic polynomial 12x^2 + 72x + 300, prove that when written
    in the form a(x+b)^2+c, where a, b, and c are constants, a + b + c = 207 -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a b c : ℝ), (∀ x, 12*x^2 + 72*x + 300 = a*(x+b)^2 + c) ∧ (a + b + c = 207) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2144_214409


namespace NUMINAMATH_CALUDE_stove_repair_ratio_l2144_214485

theorem stove_repair_ratio :
  let stove_cost : ℚ := 1200
  let total_cost : ℚ := 1400
  let wall_cost : ℚ := total_cost - stove_cost
  (wall_cost / stove_cost) = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_stove_repair_ratio_l2144_214485


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l2144_214407

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- positive sequence
  (∃ r > 0, ∀ k, a (k + 1) = r * a k) →  -- geometric sequence
  a 9 = 9 * a 7 →  -- given condition
  a m * a n = 9 * (a 1)^2 →  -- given condition
  (∀ i j : ℕ, (a i * a j = 9 * (a 1)^2) → 1/i + 9/j ≥ 1/m + 9/n) →  -- minimum condition
  1/m + 9/n = 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l2144_214407


namespace NUMINAMATH_CALUDE_arithmetic_sequence_diff_l2144_214493

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_diff (a₁ d : ℤ) :
  |arithmetic_sequence a₁ d 105 - arithmetic_sequence a₁ d 100| = 40 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_diff_l2144_214493


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2144_214443

theorem trigonometric_equation_solution (k : ℤ) :
  (∃ x : ℝ, 4 - Real.sin x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) - Real.cos (7 * x) ^ 2 = 
   Real.cos (Real.pi * k / 2021) ^ 2) ↔ 
  (∃ m : ℤ, k = 2021 * m) ∧ 
  (∀ x : ℝ, 4 - Real.sin x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) - Real.cos (7 * x) ^ 2 = 
   Real.cos (Real.pi * k / 2021) ^ 2 → 
   ∃ n : ℤ, x = Real.pi / 4 + n * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2144_214443


namespace NUMINAMATH_CALUDE_rectangle_length_is_one_point_five_times_width_l2144_214445

/-- Represents the configuration of squares and rectangles in a larger square -/
structure SquareConfiguration where
  /-- Side length of a small square -/
  s : ℝ
  /-- Length of a rectangle -/
  l : ℝ
  /-- The configuration forms a square -/
  is_square : 3 * s = 2 * l
  /-- The width of each rectangle equals the side of a small square -/
  width_eq_side : l > s

/-- Theorem stating that the length of each rectangle is 1.5 times its width -/
theorem rectangle_length_is_one_point_five_times_width (config : SquareConfiguration) :
  config.l = 1.5 * config.s := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_is_one_point_five_times_width_l2144_214445


namespace NUMINAMATH_CALUDE_lace_makers_combined_time_l2144_214462

theorem lace_makers_combined_time (t1 t2 T : ℚ) : 
  t1 = 8 → t2 = 13 → (1 / t1 + 1 / t2) * T = 1 → T = 104 / 21 := by
  sorry

end NUMINAMATH_CALUDE_lace_makers_combined_time_l2144_214462


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2144_214442

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt ((2 / x) + 2) = 3 / 2 → x = 8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2144_214442


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l2144_214451

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l2144_214451


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l2144_214413

-- Define P and Q as propositions depending on a real number x
def P (x : ℝ) : Prop := (2*x - 3)^2 < 1
def Q (x : ℝ) : Prop := x*(x - 3) < 0

-- State the theorem
theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ 
  (∃ x : ℝ, Q x ∧ ¬(P x)) := by
sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l2144_214413


namespace NUMINAMATH_CALUDE_cost_is_ten_l2144_214463

/-- Represents the cost of piano lessons -/
structure LessonCost where
  lessons_per_week : ℕ
  lesson_duration_hours : ℕ
  weeks : ℕ
  total_earnings : ℕ

/-- Calculates the cost per half-hour of teaching -/
def cost_per_half_hour (lc : LessonCost) : ℚ :=
  lc.total_earnings / (2 * lc.lessons_per_week * lc.lesson_duration_hours * lc.weeks)

/-- Theorem: The cost per half-hour of teaching is $10 -/
theorem cost_is_ten (lc : LessonCost) 
  (h1 : lc.lessons_per_week = 1)
  (h2 : lc.lesson_duration_hours = 1)
  (h3 : lc.weeks = 5)
  (h4 : lc.total_earnings = 100) : 
  cost_per_half_hour lc = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_is_ten_l2144_214463


namespace NUMINAMATH_CALUDE_focal_chord_length_l2144_214474

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 3 * x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a chord passing through the focus
structure FocalChord where
  a : PointOnParabola
  b : PointOnParabola
  passes_through_focus : True  -- We assume this property without specifying the focus

-- Theorem statement
theorem focal_chord_length 
  (ab : FocalChord) 
  (midpoint_x : ab.a.x + ab.b.x = 4) : 
  Real.sqrt ((ab.a.x - ab.b.x)^2 + (ab.a.y - ab.b.y)^2) = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_focal_chord_length_l2144_214474


namespace NUMINAMATH_CALUDE_neg_two_oplus_three_solve_equation_find_expression_value_l2144_214424

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 2 * a - a * b

-- Theorem 1
theorem neg_two_oplus_three : oplus (-2) 3 = 2 := by sorry

-- Theorem 2
theorem solve_equation (x : ℝ) : oplus (-3) x = oplus (x + 1) 5 → x = 1/2 := by sorry

-- Theorem 3
theorem find_expression_value (x y : ℝ) : oplus x 1 = 2 * (oplus 1 y) → (1/2) * x + y + 1 = 3 := by sorry

end NUMINAMATH_CALUDE_neg_two_oplus_three_solve_equation_find_expression_value_l2144_214424


namespace NUMINAMATH_CALUDE_element_order_l2144_214468

-- Define the elements as a custom type
inductive Element : Type
  | A | B | C | D | E

-- Define the properties
def in_same_period (e₁ e₂ : Element) : Prop := sorry

def forms_basic_oxide (e : Element) : Prop := sorry

def basicity (e : Element) : ℝ := sorry

def hydride_stability (e : Element) : ℝ := sorry

def ionic_radius (e : Element) : ℝ := sorry

def atomic_number (e : Element) : ℕ := sorry

-- State the theorem
theorem element_order :
  (∀ e₁ e₂ : Element, in_same_period e₁ e₂) →
  forms_basic_oxide Element.A →
  forms_basic_oxide Element.B →
  basicity Element.B > basicity Element.A →
  hydride_stability Element.C > hydride_stability Element.D →
  (∀ e : Element, ionic_radius Element.E ≤ ionic_radius e) →
  (atomic_number Element.B < atomic_number Element.A ∧
   atomic_number Element.A < atomic_number Element.E ∧
   atomic_number Element.E < atomic_number Element.D ∧
   atomic_number Element.D < atomic_number Element.C) :=
by sorry


end NUMINAMATH_CALUDE_element_order_l2144_214468


namespace NUMINAMATH_CALUDE_price_increase_percentage_l2144_214496

theorem price_increase_percentage (old_price new_price : ℝ) (h1 : old_price = 300) (h2 : new_price = 450) :
  (new_price - old_price) / old_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l2144_214496


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2144_214456

theorem arithmetic_mean_of_fractions :
  let a := 8 / 11
  let b := 9 / 11
  let c := 7 / 11
  a = (b + c) / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2144_214456


namespace NUMINAMATH_CALUDE_song_book_cost_l2144_214454

def trumpet_cost : ℝ := 149.16
def music_tool_cost : ℝ := 9.98
def total_spent : ℝ := 163.28

theorem song_book_cost :
  total_spent - (trumpet_cost + music_tool_cost) = 4.14 := by sorry

end NUMINAMATH_CALUDE_song_book_cost_l2144_214454


namespace NUMINAMATH_CALUDE_shadow_length_proportion_l2144_214408

/-- Given two objects side by side, if one object of height 20 units casts a shadow of 10 units,
    then another object of height 40 units will cast a shadow of 20 units. -/
theorem shadow_length_proportion
  (h1 : ℝ) (s1 : ℝ) (h2 : ℝ)
  (height_shadow_1 : h1 = 20)
  (shadow_1 : s1 = 10)
  (height_2 : h2 = 40)
  (proportion : h1 / s1 = h2 / (h2 / 2)) :
  h2 / 2 = 20 := by
sorry

end NUMINAMATH_CALUDE_shadow_length_proportion_l2144_214408


namespace NUMINAMATH_CALUDE_sets_A_and_B_l2144_214441

def A : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) = 0}

theorem sets_A_and_B :
  (A = {-1, 3}) ∧
  (∀ a : ℝ,
    (a = 1 ∨ a = -1 ∨ a = 3 → A ∪ B a = {-1, 1, 3}) ∧
    (a ≠ -1 ∧ a ≠ 1 ∧ a ≠ 3 → A ∪ B a = {-1, 1, 3, a}) ∧
    (a = 1 ∨ (a ≠ -1 ∧ a ≠ 1 ∧ a ≠ 3) → A ∩ B a = ∅) ∧
    (a = -1 → A ∩ B a = {-1}) ∧
    (a = 3 → A ∩ B a = {3})) :=
by sorry

end NUMINAMATH_CALUDE_sets_A_and_B_l2144_214441


namespace NUMINAMATH_CALUDE_rug_selling_price_l2144_214416

/-- Proves that the selling price per rug is $60, given the cost price, number of rugs, and total profit --/
theorem rug_selling_price 
  (cost_price : ℝ) 
  (num_rugs : ℕ) 
  (total_profit : ℝ) 
  (h1 : cost_price = 40) 
  (h2 : num_rugs = 20) 
  (h3 : total_profit = 400) : 
  (cost_price * num_rugs + total_profit) / num_rugs = 60 := by
  sorry

end NUMINAMATH_CALUDE_rug_selling_price_l2144_214416


namespace NUMINAMATH_CALUDE_five_digit_number_divisibility_l2144_214428

theorem five_digit_number_divisibility (U : ℕ) : 
  U < 10 →
  (2018 * 10 + U) % 9 = 0 →
  (2018 * 10 + U) % 8 = 3 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_number_divisibility_l2144_214428


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l2144_214417

-- Define the repeating decimals
def repeating_decimal_0_8 : ℚ := 8/9
def repeating_decimal_2_4 : ℚ := 22/9

-- State the theorem
theorem repeating_decimal_fraction :
  repeating_decimal_0_8 / repeating_decimal_2_4 = 4/11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l2144_214417


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_plus_100_l2144_214483

theorem arithmetic_series_sum_plus_100 : 
  let a₁ : ℕ := 10
  let aₙ : ℕ := 100
  let d : ℕ := 1
  let n : ℕ := (aₙ - a₁) / d + 1
  let series_sum : ℕ := n * (a₁ + aₙ) / 2
  series_sum + 100 = 5105 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_plus_100_l2144_214483


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2144_214401

theorem quadratic_root_relation (b c : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁^2 = -x₂) →
  (b^3 - 3*b*c - c^2 - c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2144_214401


namespace NUMINAMATH_CALUDE_middle_of_five_consecutive_integers_l2144_214481

/-- Given 5 consecutive integers with a sum of 60, prove that the middle number is 12 -/
theorem middle_of_five_consecutive_integers (a b c d e : ℤ) : 
  (a + b + c + d + e = 60) → 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (e = d + 1) →
  c = 12 := by
sorry

end NUMINAMATH_CALUDE_middle_of_five_consecutive_integers_l2144_214481


namespace NUMINAMATH_CALUDE_sum_of_decimal_and_fraction_l2144_214453

theorem sum_of_decimal_and_fraction : 7.31 + (1 / 5 : ℚ) = 7.51 := by sorry

end NUMINAMATH_CALUDE_sum_of_decimal_and_fraction_l2144_214453


namespace NUMINAMATH_CALUDE_complex_solution_l2144_214471

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_solution (z : ℂ) (h : determinant z 1 z (2 * Complex.I) = 3 + 2 * Complex.I) :
  z = (1 / 5 : ℂ) - (8 / 5 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_solution_l2144_214471


namespace NUMINAMATH_CALUDE_pizza_slices_per_adult_l2144_214476

theorem pizza_slices_per_adult (num_adults num_children num_pizzas slices_per_pizza slices_per_child : ℕ) :
  num_adults = 2 →
  num_children = 6 →
  num_pizzas = 3 →
  slices_per_pizza = 4 →
  slices_per_child = 1 →
  (num_pizzas * slices_per_pizza - num_children * slices_per_child) / num_adults = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_adult_l2144_214476


namespace NUMINAMATH_CALUDE_roots_of_quadratic_and_quartic_l2144_214452

theorem roots_of_quadratic_and_quartic (α β p q : ℝ) : 
  (α^2 - 3*α + 1 = 0) ∧ 
  (β^2 - 3*β + 1 = 0) ∧ 
  (α^4 - p*α^2 + q = 0) ∧ 
  (β^4 - p*β^2 + q = 0) →
  p = 7 ∧ q = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_and_quartic_l2144_214452


namespace NUMINAMATH_CALUDE_comic_book_ratio_l2144_214495

def initial_books : ℕ := 22
def final_books : ℕ := 17
def bought_books : ℕ := 6

theorem comic_book_ratio : 
  ∃ (sold_books : ℕ), 
    initial_books - sold_books + bought_books = final_books ∧
    sold_books * 2 = initial_books := by
  sorry

end NUMINAMATH_CALUDE_comic_book_ratio_l2144_214495


namespace NUMINAMATH_CALUDE_ab_minimum_value_l2144_214415

theorem ab_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 3) :
  a * b ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_ab_minimum_value_l2144_214415


namespace NUMINAMATH_CALUDE_saree_price_proof_l2144_214419

/-- Proves that given a product with two successive discounts of 10% and 5%, 
    if the final sale price is Rs. 513, then the original price was Rs. 600. -/
theorem saree_price_proof (original_price : ℝ) : 
  (original_price * (1 - 0.1) * (1 - 0.05) = 513) → original_price = 600 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_proof_l2144_214419


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l2144_214402

def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.sin θ - ρ * (Real.cos θ)^2 - Real.sin θ = 0

def cartesian_equation (x y : ℝ) : Prop :=
  x = 1 ∨ (x^2 + y^2 + y = 0 ∧ y ≠ 0)

theorem polar_to_cartesian_equivalence :
  ∀ x y ρ θ, θ ∈ Set.Ioo 0 Real.pi →
  x = ρ * Real.cos θ → y = ρ * Real.sin θ →
  (polar_equation ρ θ ↔ cartesian_equation x y) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l2144_214402


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2144_214432

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (9 - 2 * x) = 8 → x = -55 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2144_214432


namespace NUMINAMATH_CALUDE_distance_less_than_radius_l2144_214433

/-- A circle with center O and radius 3 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A point P inside the circle -/
structure PointInside (c : Circle) :=
  (P : ℝ × ℝ)
  (h_inside : dist P c.O < c.radius)

/-- Theorem: The distance between the center and a point inside the circle is less than 3 -/
theorem distance_less_than_radius (c : Circle) (p : PointInside c) :
  dist p.P c.O < 3 := by sorry

end NUMINAMATH_CALUDE_distance_less_than_radius_l2144_214433


namespace NUMINAMATH_CALUDE_video_upload_total_l2144_214430

theorem video_upload_total (days_in_month : ℕ) (initial_daily_upload : ℕ) : 
  days_in_month = 30 →
  initial_daily_upload = 10 →
  (days_in_month / 2 * initial_daily_upload) + 
  (days_in_month / 2 * (2 * initial_daily_upload)) = 450 := by
sorry

end NUMINAMATH_CALUDE_video_upload_total_l2144_214430


namespace NUMINAMATH_CALUDE_units_digit_product_l2144_214434

theorem units_digit_product (n : ℕ) : 2^2023 * 5^2024 * 11^2025 ≡ 0 [MOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l2144_214434


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l2144_214400

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ is_two_digit p ∧ (p ∣ binomial_coefficient 300 150) ∧
  ∀ (q : ℕ), Nat.Prime q → is_two_digit q → (q ∣ binomial_coefficient 300 150) → q ≤ p :=
by
  use 89
  sorry

#check largest_two_digit_prime_factor

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l2144_214400


namespace NUMINAMATH_CALUDE_fraction_inequality_l2144_214488

theorem fraction_inequality (x : ℝ) : (4 * x) / (x^2 + 4) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2144_214488


namespace NUMINAMATH_CALUDE_am_gm_and_specific_case_l2144_214497

theorem am_gm_and_specific_case :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → (a + b + c) / 3 ≥ (a * b * c) ^ (1/3)) ∧
  ((4 + 9 + 16) / 3 - (4 * 9 * 16) ^ (1/3) ≠ 1) := by
sorry

end NUMINAMATH_CALUDE_am_gm_and_specific_case_l2144_214497


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l2144_214431

def total_balls : ℕ := 9
def white_balls : ℕ := 4
def black_balls : ℕ := 5
def drawn_balls : ℕ := 3

theorem ball_drawing_theorem :
  -- 1. Total number of ways to choose 3 balls from 9 balls
  Nat.choose total_balls drawn_balls = 84 ∧
  -- 2. Number of ways to choose 2 white and 1 black
  (Nat.choose white_balls 2 * Nat.choose black_balls 1) = 30 ∧
  -- 3. Number of ways to choose at least 2 white balls
  (Nat.choose white_balls 2 * Nat.choose black_balls 1 + Nat.choose white_balls 3) = 34 ∧
  -- 4. Probability of choosing 2 white and 1 black
  (↑(Nat.choose white_balls 2 * Nat.choose black_balls 1) / ↑(Nat.choose total_balls drawn_balls) : ℚ) = 30 / 84 ∧
  -- 5. Probability of choosing at least 2 white balls
  (↑(Nat.choose white_balls 2 * Nat.choose black_balls 1 + Nat.choose white_balls 3) / ↑(Nat.choose total_balls drawn_balls) : ℚ) = 34 / 84 :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l2144_214431


namespace NUMINAMATH_CALUDE_elevator_max_weight_elevator_problem_l2144_214479

/-- Calculates the maximum weight of the next person to enter an elevator without overloading it. -/
theorem elevator_max_weight (num_adults : ℕ) (num_children : ℕ) (avg_adult_weight : ℝ) 
  (avg_child_weight : ℝ) (original_capacity : ℝ) (capacity_increase : ℝ) : ℝ :=
  let total_adult_weight := num_adults * avg_adult_weight
  let total_child_weight := num_children * avg_child_weight
  let current_weight := total_adult_weight + total_child_weight
  let new_capacity := original_capacity * (1 + capacity_increase)
  new_capacity - current_weight

/-- Proves that the maximum weight of the next person to enter the elevator is 250 pounds. -/
theorem elevator_problem : 
  elevator_max_weight 7 5 150 70 1500 0.1 = 250 := by
  sorry

end NUMINAMATH_CALUDE_elevator_max_weight_elevator_problem_l2144_214479


namespace NUMINAMATH_CALUDE_expression_equality_equation_solutions_l2144_214403

-- Problem 1
theorem expression_equality : 
  |Real.sqrt 3 - 1| - 2 * Real.cos (π / 3) + (Real.sqrt 3 - 2)^2 + Real.sqrt 12 = 5 - Real.sqrt 3 := by
  sorry

-- Problem 2
theorem equation_solutions (x : ℝ) : 
  2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_equation_solutions_l2144_214403


namespace NUMINAMATH_CALUDE_line_slope_l2144_214469

/-- The slope of the line given by the equation 4y + 5x = 20 is -5/4 -/
theorem line_slope (x y : ℝ) : 4 * y + 5 * x = 20 → (y - 5) / (-5 / 4) = x := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2144_214469


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_9_l2144_214487

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_9_l2144_214487


namespace NUMINAMATH_CALUDE_quadratic_roots_k_value_l2144_214480

theorem quadratic_roots_k_value (k : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 10 * x + k = 0 ↔ x = 5 + Real.sqrt 15 ∨ x = 5 - Real.sqrt 15) →
  k = 85 / 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_k_value_l2144_214480


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2144_214411

theorem complex_equation_solution (z : ℂ) : z * Complex.I = -1 + (3/4) * Complex.I → z = 3/4 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2144_214411


namespace NUMINAMATH_CALUDE_xaxaxa_divisible_by_seven_l2144_214461

theorem xaxaxa_divisible_by_seven (X A : ℕ) 
  (h_digits : X < 10 ∧ A < 10) 
  (h_distinct : X ≠ A) : 
  ∃ k : ℕ, 101010 * X + 10101 * A = 7 * k := by
sorry

end NUMINAMATH_CALUDE_xaxaxa_divisible_by_seven_l2144_214461


namespace NUMINAMATH_CALUDE_equation_solution_l2144_214490

theorem equation_solution : 
  let x₁ : ℝ := (3 + Real.sqrt 17) / 2
  let x₂ : ℝ := (-3 - Real.sqrt 17) / 2
  (x₁^2 - 3 * |x₁| - 2 = 0) ∧ 
  (x₂^2 - 3 * |x₂| - 2 = 0) ∧ 
  (∀ x : ℝ, x^2 - 3 * |x| - 2 = 0 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2144_214490


namespace NUMINAMATH_CALUDE_num_arrangements_eq_162_l2144_214467

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items --/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of different arrangements for dispatching volunteers --/
def num_arrangements : ℕ :=
  let total_volunteers := 5
  let dispatched_volunteers := 4
  let num_communities := 3
  let scenario1 := choose 3 2 * (choose 4 2 - 1) * arrange 3 3
  let scenario2 := choose 2 1 * choose 4 2 * arrange 3 3
  scenario1 + scenario2

theorem num_arrangements_eq_162 : num_arrangements = 162 := by sorry

end NUMINAMATH_CALUDE_num_arrangements_eq_162_l2144_214467


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2144_214412

theorem quadratic_one_solution (m : ℚ) :
  (∃! x, 3 * x^2 - 6 * x + m = 0) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2144_214412


namespace NUMINAMATH_CALUDE_collinear_vectors_m_equals_six_l2144_214422

/-- Two vectors are collinear if the determinant of their components is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given plane vectors a and b, if they are collinear, then m = 6 -/
theorem collinear_vectors_m_equals_six :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-3, m)
  collinear a b → m = 6 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_equals_six_l2144_214422


namespace NUMINAMATH_CALUDE_georges_walk_to_school_l2144_214410

/-- Proves that given the conditions of George's walk to school, 
    the speed required for the second mile to arrive on time is 6 mph. -/
theorem georges_walk_to_school (total_distance : ℝ) (normal_speed : ℝ) 
  (normal_time : ℝ) (first_mile_speed : ℝ) :
  total_distance = 2 →
  normal_speed = 4 →
  normal_time = 0.5 →
  first_mile_speed = 3 →
  ∃ (second_mile_speed : ℝ),
    second_mile_speed = 6 ∧
    (1 / first_mile_speed + 1 / second_mile_speed = normal_time) :=
by sorry

end NUMINAMATH_CALUDE_georges_walk_to_school_l2144_214410


namespace NUMINAMATH_CALUDE_find_a_value_l2144_214472

def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

theorem find_a_value : ∃ a : ℝ, (B a ⊆ A a) ∧ (a = -1 ∨ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l2144_214472


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l2144_214440

/-- Proves that the slower speed is 10 km/hr given the conditions of the problem -/
theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (extra_distance : ℝ)
  (h1 : actual_distance = 100)
  (h2 : faster_speed = 12)
  (h3 : extra_distance = 20)
  (slower_speed : ℝ)
  (h4 : faster_speed * (actual_distance / slower_speed) = actual_distance + extra_distance) :
  slower_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l2144_214440


namespace NUMINAMATH_CALUDE_smallest_non_odd_ending_digit_l2144_214438

def is_odd_ending_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def is_digit (n : ℕ) : Prop :=
  n ≤ 9

theorem smallest_non_odd_ending_digit :
  ∀ d : ℕ, is_digit d → 
    (¬is_odd_ending_digit d → d ≥ 0) ∧
    (∀ d' : ℕ, is_digit d' → ¬is_odd_ending_digit d' → d ≤ d') :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_odd_ending_digit_l2144_214438


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2144_214447

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-3/5, 4/5), prove that cos(α) = -3/5 -/
theorem cos_alpha_value (α : Real) (h1 : ∃ (x y : Real), x = -3/5 ∧ y = 4/5 ∧ 
  (Real.cos α = x ∧ Real.sin α = y)) : Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2144_214447


namespace NUMINAMATH_CALUDE_video_game_spending_l2144_214499

theorem video_game_spending (weekly_allowance : ℝ) (weeks : ℕ) 
  (video_game_cost : ℝ) (book_fraction : ℝ) (remaining : ℝ) :
  weekly_allowance = 10 →
  weeks = 4 →
  book_fraction = 1/4 →
  remaining = 15 →
  video_game_cost > 0 →
  video_game_cost < weekly_allowance * weeks →
  remaining = weekly_allowance * weeks - video_game_cost - 
    (weekly_allowance * weeks - video_game_cost) * book_fraction →
  video_game_cost / (weekly_allowance * weeks) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_video_game_spending_l2144_214499


namespace NUMINAMATH_CALUDE_blakes_change_is_correct_l2144_214450

/-- Calculates the change Blake receives after buying candy with discounts -/
def blakes_change (lollipop_price : ℚ) (gummy_price : ℚ) (candy_bar_price : ℚ) : ℚ :=
  let chocolate_price := 4 * lollipop_price
  let lollipop_cost := 3 * lollipop_price + lollipop_price / 2
  let chocolate_cost := 4 * chocolate_price + 2 * (chocolate_price * 3 / 4)
  let gummy_cost := 3 * gummy_price
  let candy_bar_cost := 5 * candy_bar_price
  let total_cost := lollipop_cost + chocolate_cost + gummy_cost + candy_bar_cost
  let total_given := 4 * 20 + 2 * 5 + 5 * 1
  total_given - total_cost

/-- Theorem stating that Blake's change is $27.50 -/
theorem blakes_change_is_correct :
  blakes_change 2 3 (3/2) = 55/2 := by sorry

end NUMINAMATH_CALUDE_blakes_change_is_correct_l2144_214450


namespace NUMINAMATH_CALUDE_garden_perimeter_l2144_214421

theorem garden_perimeter :
  ∀ (length breadth perimeter : ℝ),
    length = 258 →
    breadth = 82 →
    perimeter = 2 * (length + breadth) →
    perimeter = 680 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l2144_214421


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2144_214489

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2144_214489


namespace NUMINAMATH_CALUDE_dolphin_training_hours_l2144_214475

theorem dolphin_training_hours 
  (num_dolphins : ℕ) 
  (num_trainers : ℕ) 
  (hours_per_trainer : ℕ) 
  (h1 : num_dolphins = 4) 
  (h2 : num_trainers = 2) 
  (h3 : hours_per_trainer = 6) : 
  (num_trainers * hours_per_trainer) / num_dolphins = 3 := by
sorry

end NUMINAMATH_CALUDE_dolphin_training_hours_l2144_214475


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2144_214458

theorem sum_of_squares_16_to_30 (sum_1_to_15 : ℕ) (sum_1_to_30 : ℕ) : 
  sum_1_to_15 = 1240 → 
  sum_1_to_30 = (30 * 31 * 61) / 6 →
  sum_1_to_30 - sum_1_to_15 = 8215 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2144_214458


namespace NUMINAMATH_CALUDE_opposite_sides_range_l2144_214457

/-- Given that the origin (0, 0) and the point (1, 1) are on opposite sides of the line x + y - a = 0,
    prove that the range of values for a is (0, 2) -/
theorem opposite_sides_range (a : ℝ) : 
  (∀ (x y : ℝ), x + y - a = 0 → (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
  (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l2144_214457
