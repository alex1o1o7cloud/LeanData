import Mathlib

namespace arithmetic_sequence_difference_l2648_264842

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  positive : ∀ n, a n > 0
  initial : a 1 = 1
  geometric : (a 3) * (a 11) = (a 4 + 5/2)^2
  arithmetic : ∀ n m, a (n + 1) - a n = a (m + 1) - a m

/-- The theorem to be proved -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) (m n : ℕ) 
  (h : m - n = 8) : seq.a m - seq.a n = 12 := by
  sorry

end arithmetic_sequence_difference_l2648_264842


namespace ball_max_height_l2648_264825

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 16

/-- Theorem stating that the maximum height of the ball is 141 feet -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 141 :=
sorry

end ball_max_height_l2648_264825


namespace prob_product_multiple_of_four_l2648_264821

/-- A fair 10-sided die -/
def decagonal_die := Finset.range 10

/-- A fair 12-sided die -/
def dodecagonal_die := Finset.range 12

/-- The probability of an event occurring when rolling a fair n-sided die -/
def prob (event : Finset ℕ) (die : Finset ℕ) : ℚ :=
  (event ∩ die).card / die.card

/-- The event of rolling a multiple of 4 -/
def multiple_of_four (die : Finset ℕ) : Finset ℕ :=
  die.filter (fun x => x % 4 = 0)

/-- The probability that the product of rolls from a 10-sided die and a 12-sided die is a multiple of 4 -/
theorem prob_product_multiple_of_four :
  prob (multiple_of_four decagonal_die) decagonal_die +
  prob (multiple_of_four dodecagonal_die) dodecagonal_die -
  prob (multiple_of_four decagonal_die) decagonal_die *
  prob (multiple_of_four dodecagonal_die) dodecagonal_die = 2/5 := by
  sorry

end prob_product_multiple_of_four_l2648_264821


namespace car_rental_cost_l2648_264864

theorem car_rental_cost (total_cost : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) :
  total_cost = 46.12 ∧ 
  miles_driven = 214 ∧ 
  cost_per_mile = 0.08 →
  ∃ daily_rental_cost : ℝ, 
    daily_rental_cost = 29 ∧ 
    total_cost = daily_rental_cost + miles_driven * cost_per_mile :=
by sorry

end car_rental_cost_l2648_264864


namespace meeting_point_theorem_l2648_264885

/-- Represents the circular path and the walkers' characteristics -/
structure CircularPath where
  totalBlocks : ℕ
  janeSpeedMultiplier : ℕ

/-- Represents the distance walked by each person when they meet -/
structure MeetingPoint where
  hectorDistance : ℕ
  janeDistance : ℕ

/-- Calculates the meeting point given a circular path -/
def calculateMeetingPoint (path : CircularPath) : MeetingPoint :=
  sorry

/-- Theorem stating that Hector walks 6 blocks when they meet -/
theorem meeting_point_theorem (path : CircularPath) 
  (h1 : path.totalBlocks = 24)
  (h2 : path.janeSpeedMultiplier = 3) :
  (calculateMeetingPoint path).hectorDistance = 6 :=
  sorry

end meeting_point_theorem_l2648_264885


namespace binomial_coefficient_equality_l2648_264806

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose n 3 = Nat.choose (n - 1) 3 + Nat.choose (n - 1) 4) → n = 7 := by
  sorry

end binomial_coefficient_equality_l2648_264806


namespace half_inequality_l2648_264897

theorem half_inequality (a b : ℝ) (h : a > b) : a/2 > b/2 := by
  sorry

end half_inequality_l2648_264897


namespace inequality_proof_l2648_264863

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + 3*c) / (3*a + 3*b + 2*c) + 
  (a + 3*b + c) / (3*a + 2*b + 3*c) + 
  (3*a + b + c) / (2*a + 3*b + 3*c) ≥ 15/8 := by
  sorry

end inequality_proof_l2648_264863


namespace solve_problem_l2648_264820

/-- The number of Adidas shoes Alice sold to meet her quota -/
def problem : Prop :=
  let quota : ℕ := 1000
  let adidas_price : ℕ := 45
  let nike_price : ℕ := 60
  let reebok_price : ℕ := 35
  let nike_sold : ℕ := 8
  let reebok_sold : ℕ := 9
  let above_goal : ℕ := 65
  ∃ adidas_sold : ℕ,
    adidas_sold * adidas_price + nike_sold * nike_price + reebok_sold * reebok_price = quota + above_goal ∧
    adidas_sold = 6

theorem solve_problem : problem := by
  sorry

end solve_problem_l2648_264820


namespace intersection_of_M_and_N_l2648_264828

def M : Set ℤ := {-1, 0, 1, 2}

def f (x : ℤ) : ℤ := Int.natAbs x

def N : Set ℤ := f '' M

theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} := by
  sorry

end intersection_of_M_and_N_l2648_264828


namespace no_bounded_function_satisfying_inequality_l2648_264814

theorem no_bounded_function_satisfying_inequality :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, ∃ M : ℝ, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y : ℝ, f (x + y)^2 ≥ f x^2 + 2 * f (x * y) + f y^2) :=
by sorry

end no_bounded_function_satisfying_inequality_l2648_264814


namespace container_volume_ratio_l2648_264867

theorem container_volume_ratio : 
  ∀ (C D : ℚ), C > 0 → D > 0 → 
  (3 / 4 : ℚ) * C = (5 / 8 : ℚ) * D → 
  C / D = (5 / 6 : ℚ) := by
  sorry

end container_volume_ratio_l2648_264867


namespace integer_x_is_seven_l2648_264862

theorem integer_x_is_seven (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : 9 > x ∧ x > 6)
  (h4 : 8 > x ∧ x > 0)
  (h5 : x + 1 < 9) :
  x = 7 := by
  sorry

end integer_x_is_seven_l2648_264862


namespace prob_12th_roll_last_l2648_264859

/-- The number of sides on the die -/
def n : ℕ := 8

/-- The number of rolls -/
def r : ℕ := 12

/-- The probability of rolling the same number on the rth roll as on the (r-1)th roll -/
def p_same : ℚ := 1 / n

/-- The probability of rolling a different number on the rth roll from the (r-1)th roll -/
def p_diff : ℚ := (n - 1) / n

/-- The probability that the rth roll is the last roll in the sequence -/
def prob_last_roll (r : ℕ) : ℚ := p_diff^(r - 2) * p_same

theorem prob_12th_roll_last :
  prob_last_roll r = (n - 1)^(r - 2) / n^r := by sorry

end prob_12th_roll_last_l2648_264859


namespace pizza_ingredients_calculation_l2648_264838

/-- Pizza ingredients calculation -/
theorem pizza_ingredients_calculation 
  (water : ℕ) 
  (flour : ℕ) 
  (salt : ℚ) 
  (h1 : water = 10)
  (h2 : flour = 16)
  (h3 : salt = (1/2 : ℚ) * flour) :
  (water + flour : ℕ) = 26 ∧ salt = 8 := by
  sorry

end pizza_ingredients_calculation_l2648_264838


namespace arithmetic_progression_difference_l2648_264831

theorem arithmetic_progression_difference (x y z k : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 → k ≠ 0 → x ≠ y → y ≠ z → x ≠ z →
  ∃ d : ℝ, (y * (z - x) + k) - (x * (y - z) + k) = d ∧
           (z * (x - y) + k) - (y * (z - x) + k) = d →
  d = 0 :=
sorry

end arithmetic_progression_difference_l2648_264831


namespace custom_equation_solution_l2648_264884

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- State the theorem
theorem custom_equation_solution :
  ∀ x : ℝ, star 3 x = 27 → x = 6 := by
  sorry

end custom_equation_solution_l2648_264884


namespace parallel_vectors_m_value_l2648_264803

/-- 
Given two parallel vectors a and b in ℝ², where a = (-2, 1) and b = (1, m),
prove that m = -1/2.
-/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (-2, 1)) 
  (h2 : b = (1, m)) 
  (h3 : ∃ (k : ℝ), a = k • b) : 
  m = -1/2 := by
sorry

end parallel_vectors_m_value_l2648_264803


namespace quadratic_roots_property_l2648_264801

theorem quadratic_roots_property (a b : ℝ) : 
  a^2 - 2*a - 1 = 0 → b^2 - 2*b - 1 = 0 → a^2 + 2*b - a*b = 6 := by
  sorry

end quadratic_roots_property_l2648_264801


namespace three_people_seven_steps_l2648_264829

/-- The number of ways to arrange n people on m steps with at most k people per step -/
def arrange (n m k : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange 3 people on 7 steps with at most 2 people per step -/
theorem three_people_seven_steps : arrange 3 7 2 = 336 := by
  sorry

end three_people_seven_steps_l2648_264829


namespace rectangle_area_ratio_l2648_264899

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a/c = b/d = 2/5, then the ratio of A's area to B's area is 4:25. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 5) (h2 : b / d = 2 / 5) :
  (a * b) / (c * d) = 4 / 25 := by
  sorry

end rectangle_area_ratio_l2648_264899


namespace total_plans_is_180_l2648_264811

def male_teachers : ℕ := 4
def female_teachers : ℕ := 3
def schools : ℕ := 3

-- Function to calculate the number of ways to select and assign teachers
def selection_and_assignment_plans : ℕ :=
  (male_teachers.choose 1 * female_teachers.choose 2 +
   male_teachers.choose 2 * female_teachers.choose 1) *
  schools.factorial

-- Theorem to prove
theorem total_plans_is_180 :
  selection_and_assignment_plans = 180 := by
  sorry

end total_plans_is_180_l2648_264811


namespace inscribed_triangle_polygon_sides_l2648_264841

/-- A triangle inscribed in a circle with specific angle relationships -/
structure InscribedTriangle where
  -- The circle in which the triangle is inscribed
  circle : Real
  -- The angles of the triangle
  angleA : Real
  angleB : Real
  angleC : Real
  -- The number of sides of the regular polygon
  n : ℕ
  -- Conditions
  angle_sum : angleA + angleB + angleC = 180
  angle_B : angleB = 3 * angleA
  angle_C : angleC = 5 * angleA
  polygon_arc : (360 : Real) / n = 140

/-- Theorem: The number of sides of the regular polygon is 5 -/
theorem inscribed_triangle_polygon_sides (t : InscribedTriangle) : t.n = 5 := by
  sorry

end inscribed_triangle_polygon_sides_l2648_264841


namespace rectangle_area_l2648_264832

/-- Given a rectangle where the length is four times the width and the perimeter is 200 cm,
    prove that its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 8 * w + 2 * w = 200) : w * (4 * w) = 1600 := by
  sorry

end rectangle_area_l2648_264832


namespace factorial_difference_l2648_264839

theorem factorial_difference (n : ℕ) (h : n.factorial = 362880) : 
  (n + 1).factorial - n.factorial = 3265920 := by
  sorry

end factorial_difference_l2648_264839


namespace smallest_prime_divisor_of_sum_l2648_264858

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → Prop) :
  (∀ n : ℕ, p n 2 → (∃ m : ℕ, n = 2 * m)) →
  (∀ n : ℕ, p 2 n → n = 2 ∨ n > 2) →
  p (3^20 + 11^14) 2 ∧ ∀ q : ℕ, p (3^20 + 11^14) q → q ≥ 2 :=
sorry

end smallest_prime_divisor_of_sum_l2648_264858


namespace necklace_labeling_theorem_l2648_264879

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

structure Necklace :=
  (beads : ℕ)

def valid_labeling (n : ℕ) (A B : Necklace) (labeling : ℕ → ℕ) : Prop :=
  (∀ i, i < A.beads + B.beads → n ≤ labeling i ∧ labeling i ≤ n + 32) ∧
  (∀ i j, i ≠ j → i < A.beads + B.beads → j < A.beads + B.beads → labeling i ≠ labeling j) ∧
  (∀ i, i < A.beads - 1 → is_coprime (labeling i) (labeling (i + 1))) ∧
  (is_coprime (labeling 0) (labeling (A.beads - 1))) ∧
  (∀ i, A.beads ≤ i ∧ i < A.beads + B.beads - 1 → is_coprime (labeling i) (labeling (i + 1))) ∧
  (is_coprime (labeling A.beads) (labeling (A.beads + B.beads - 1)))

theorem necklace_labeling_theorem (n : ℕ) (A B : Necklace) 
  (h_n_odd : is_odd n) (h_n_ge_1 : n ≥ 1) (h_A : A.beads = 14) (h_B : B.beads = 19) :
  ∃ labeling : ℕ → ℕ, valid_labeling n A B labeling :=
sorry

end necklace_labeling_theorem_l2648_264879


namespace arithmetic_mean_increase_l2648_264868

theorem arithmetic_mean_increase (a b c d e : ℝ) :
  let original_set := [a, b, c, d, e]
  let new_set := original_set.map (λ x => x + 15)
  let original_mean := (a + b + c + d + e) / 5
  let new_mean := (new_set.sum) / 5
  new_mean = original_mean + 15 := by
sorry

end arithmetic_mean_increase_l2648_264868


namespace rectangle_dimensions_l2648_264866

theorem rectangle_dimensions (x y : ℝ) : 
  (2*x + y) * (2*y) = 90 ∧ x*y = 10 → x = 2 ∧ y = 5 := by sorry

end rectangle_dimensions_l2648_264866


namespace f_properties_l2648_264845

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_properties :
  ∃ (x₀ : ℝ),
    (∀ x > 0, HasDerivAt f (Real.log x + 1) x) ∧
    (HasDerivAt f 2 x₀ → x₀ = Real.exp 1) ∧
    (∀ x ≥ Real.exp (-1), StrictMono f) ∧
    (∃! p, f p = -Real.exp (-1)) :=
by sorry

end f_properties_l2648_264845


namespace sons_age_l2648_264849

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end sons_age_l2648_264849


namespace inverse_g_150_l2648_264889

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^4 + 6

-- State the theorem
theorem inverse_g_150 : 
  g ((2 : ℝ) * (3 : ℝ)^(1/4)) = 150 :=
by sorry

end inverse_g_150_l2648_264889


namespace triangle_angle_theorem_l2648_264808

theorem triangle_angle_theorem (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < B → B < π →
  a^2 + c^2 = b^2 + a*c →
  B = π/3 := by
  sorry

end triangle_angle_theorem_l2648_264808


namespace rectangle_area_perimeter_relation_l2648_264860

/-- Given a rectangle with length 4x inches and width 3x + 4 inches,
    where its area is twice its perimeter, prove that x = 1. -/
theorem rectangle_area_perimeter_relation (x : ℝ) : 
  (4 * x) * (3 * x + 4) = 2 * (2 * (4 * x) + 2 * (3 * x + 4)) → x = 1 := by
  sorry

end rectangle_area_perimeter_relation_l2648_264860


namespace lcm_of_16_24_45_l2648_264871

theorem lcm_of_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end lcm_of_16_24_45_l2648_264871


namespace a_less_than_one_l2648_264873

-- Define the function f
def f (x : ℝ) : ℝ := -x^5 - 3*x^3 - 5*x + 3

-- State the theorem
theorem a_less_than_one (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 := by
  sorry

end a_less_than_one_l2648_264873


namespace sqrt_17_minus_1_gt_3_l2648_264834

theorem sqrt_17_minus_1_gt_3 : Real.sqrt 17 - 1 > 3 := by
  sorry

end sqrt_17_minus_1_gt_3_l2648_264834


namespace log_inequality_l2648_264856

theorem log_inequality (a b c : ℝ) : 
  a = Real.log 3 / Real.log 4 →
  b = Real.log 4 / Real.log 3 →
  c = Real.log 3 / Real.log 5 →
  b > a ∧ a > c :=
by sorry

end log_inequality_l2648_264856


namespace prize_probability_l2648_264847

theorem prize_probability (odds_favorable : ℕ) (odds_unfavorable : ℕ) 
  (h_odds : odds_favorable = 5 ∧ odds_unfavorable = 6) :
  let total_outcomes := odds_favorable + odds_unfavorable
  let prob_not_prize := odds_unfavorable / total_outcomes
  (prob_not_prize ^ 2 : ℚ) = 36 / 121 :=
by sorry

end prize_probability_l2648_264847


namespace square_difference_equality_l2648_264857

theorem square_difference_equality : 1012^2 - 992^2 - 1009^2 + 995^2 = 12024 := by
  sorry

end square_difference_equality_l2648_264857


namespace probability_not_exceeding_60W_l2648_264895

def total_bulbs : ℕ := 250
def bulbs_100W : ℕ := 100
def bulbs_60W : ℕ := 50
def bulbs_25W : ℕ := 50
def bulbs_15W : ℕ := 50

theorem probability_not_exceeding_60W :
  let p := (bulbs_60W + bulbs_25W + bulbs_15W) / total_bulbs
  p = 3/5 := by sorry

end probability_not_exceeding_60W_l2648_264895


namespace five_plumbers_three_areas_l2648_264875

/-- The number of ways to assign plumbers to residential areas. -/
def assignment_plans (n_plumbers : ℕ) (n_areas : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating the number of assignment plans for 5 plumbers and 3 areas. -/
theorem five_plumbers_three_areas : 
  assignment_plans 5 3 = 150 := by
  sorry

end five_plumbers_three_areas_l2648_264875


namespace exactly_one_female_probability_l2648_264852

def total_students : ℕ := 50
def male_students : ℕ := 30
def female_students : ℕ := 20
def group_size : ℕ := 5

def male_in_group : ℕ := male_students * group_size / total_students
def female_in_group : ℕ := female_students * group_size / total_students

theorem exactly_one_female_probability : 
  (male_in_group * female_in_group * 2) / (group_size * (group_size - 1)) = 3/5 :=
by sorry

end exactly_one_female_probability_l2648_264852


namespace positive_solution_of_equation_l2648_264817

theorem positive_solution_of_equation (x : ℝ) :
  x > 0 ∧ x + 17 = 60 * (1/x) → x = 3 := by
  sorry

end positive_solution_of_equation_l2648_264817


namespace complex_number_properties_l2648_264865

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 5*m)

theorem complex_number_properties :
  (∃ m : ℝ, z m = Complex.I * (z m).im) ∧
  (∃ m : ℝ, z m = 3 + 6*Complex.I) ∧
  (∃ m : ℝ, 0 < m ∧ m < 3 ∧ (z m).re > 0 ∧ (z m).im < 0) :=
by sorry


end complex_number_properties_l2648_264865


namespace manuscript_cost_theorem_l2648_264824

/-- Calculates the total cost of typing and revising a manuscript --/
def manuscript_cost (total_pages : ℕ) (once_revised : ℕ) (twice_revised : ℕ) 
  (initial_rate : ℕ) (revision_rate : ℕ) : ℕ :=
  let not_revised := total_pages - once_revised - twice_revised
  let initial_cost := total_pages * initial_rate
  let once_revised_cost := once_revised * revision_rate
  let twice_revised_cost := twice_revised * (2 * revision_rate)
  initial_cost + once_revised_cost + twice_revised_cost

/-- Theorem stating the total cost of the manuscript --/
theorem manuscript_cost_theorem :
  manuscript_cost 200 80 20 5 3 = 1360 := by
  sorry

end manuscript_cost_theorem_l2648_264824


namespace medal_award_ways_l2648_264823

-- Define the total number of sprinters
def total_sprinters : ℕ := 10

-- Define the number of Spanish sprinters
def spanish_sprinters : ℕ := 4

-- Define the number of medals
def medals : ℕ := 3

-- Function to calculate the number of ways to award medals
def award_medals : ℕ := sorry

-- Theorem statement
theorem medal_award_ways :
  award_medals = 696 :=
sorry

end medal_award_ways_l2648_264823


namespace steve_calculation_l2648_264844

theorem steve_calculation (x : ℝ) : (x / 8) - 20 = 12 → (x * 8) + 20 = 2068 := by
  sorry

end steve_calculation_l2648_264844


namespace sufficient_not_necessary_condition_l2648_264870

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the set {1, 2}
def set_1_2 : Set ℝ := {1, 2}

-- Theorem statement
theorem sufficient_not_necessary_condition :
  (∀ m ∈ set_1_2, log10 m < 1) ∧
  (∃ m : ℝ, log10 m < 1 ∧ m ∉ set_1_2) := by
  sorry

end sufficient_not_necessary_condition_l2648_264870


namespace sum_equals_80790_l2648_264881

theorem sum_equals_80790 : 30 + 80000 + 700 + 60 = 80790 := by
  sorry

end sum_equals_80790_l2648_264881


namespace min_value_of_a_l2648_264816

theorem min_value_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0) (hxy : x ≠ y)
  (h : (2 * x - y / Real.exp 1) * Real.log (y / x) = x / (a * Real.exp 1)) :
  a ≥ 1 / Real.exp 1 := by
sorry

end min_value_of_a_l2648_264816


namespace cube_volume_from_surface_area_l2648_264822

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 294 → s^3 = 343 :=
by
  sorry

end cube_volume_from_surface_area_l2648_264822


namespace distribute_six_books_three_people_l2648_264888

/-- The number of ways to distribute n different books among k people, 
    with each person getting at least 1 book -/
def distribute_books (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 6 different books among 3 people, 
    with each person getting at least 1 book, can be done in 540 ways -/
theorem distribute_six_books_three_people : 
  distribute_books 6 3 = 540 := by sorry

end distribute_six_books_three_people_l2648_264888


namespace math_players_count_central_park_school_math_players_l2648_264830

theorem math_players_count (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) : ℕ :=
  let math_players := total_players - (physics_players - both_subjects)
  math_players

theorem central_park_school_math_players : 
  math_players_count 15 10 4 = 9 := by sorry

end math_players_count_central_park_school_math_players_l2648_264830


namespace johnny_table_legs_l2648_264878

/-- Given the number of tables, planks per surface, and total planks,
    calculate the number of planks needed for the legs of each table. -/
def planksForLegs (numTables : ℕ) (planksPerSurface : ℕ) (totalPlanks : ℕ) : ℕ :=
  (totalPlanks - numTables * planksPerSurface) / numTables

/-- Theorem stating that given the specific values in the problem,
    the number of planks needed for the legs of each table is 4. -/
theorem johnny_table_legs :
  planksForLegs 5 5 45 = 4 := by
  sorry

end johnny_table_legs_l2648_264878


namespace hyperbola_equation_l2648_264807

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the condition that the hyperbola passes through (2, 1)
def passes_through_point (a b : ℝ) : Prop := hyperbola a b 2 1

-- Define the condition that the hyperbola and ellipse share the same foci
def same_foci (a b : ℝ) : Prop := a^2 + b^2 = 3

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) 
  (h1 : passes_through_point a b) 
  (h2 : same_foci a b) : 
  ∀ x y : ℝ, hyperbola 2 1 x y := by
  sorry

end hyperbola_equation_l2648_264807


namespace order_of_magnitude_l2648_264872

noncomputable def a : ℝ := Real.exp (Real.exp 1)
noncomputable def b : ℝ := Real.pi ^ Real.pi
noncomputable def c : ℝ := Real.exp Real.pi
noncomputable def d : ℝ := Real.pi ^ (Real.exp 1)

theorem order_of_magnitude : a < d ∧ d < c ∧ c < b := by sorry

end order_of_magnitude_l2648_264872


namespace polynomial_properties_l2648_264882

/-- The polynomial -8x³y^(m+1) + xy² - 3/4x³ + 6y is a sixth-degree quadrinomial -/
def is_sixth_degree_quadrinomial (m : ℕ) : Prop :=
  3 + (m + 1) = 6

/-- The monomial 2/5πx^ny^(5-m) has the same degree as the polynomial -/
def monomial_same_degree (m n : ℕ) : Prop :=
  n + (5 - m) = 6

/-- The polynomial coefficients sum to -7/4 -/
def coefficients_sum : ℚ :=
  -8 + 1 + (-3/4) + 6

theorem polynomial_properties :
  ∃ (m n : ℕ),
    is_sixth_degree_quadrinomial m ∧
    monomial_same_degree m n ∧
    m = 2 ∧
    n = 3 ∧
    coefficients_sum = -7/4 := by sorry

end polynomial_properties_l2648_264882


namespace intersection_A_complement_B_l2648_264896

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -x^2 + 2*x + 3 > 0}
def B : Set ℝ := {x : ℝ | x - 2 < 0}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = Set.Icc 2 3 := by sorry

end intersection_A_complement_B_l2648_264896


namespace workshop_attendance_workshop_attendance_proof_l2648_264809

theorem workshop_attendance : ℕ → Prop :=
  fun total =>
    ∃ (wolf nobel wolf_nobel non_wolf_nobel non_wolf_non_nobel : ℕ),
      -- Total Wolf Prize laureates
      wolf = 31 ∧
      -- Wolf Prize laureates who are also Nobel Prize laureates
      wolf_nobel = 18 ∧
      -- Total Nobel Prize laureates
      nobel = 29 ∧
      -- Difference between Nobel (non-Wolf) and non-Nobel (non-Wolf)
      non_wolf_nobel = non_wolf_non_nobel + 3 ∧
      -- Total scientists is sum of all categories
      total = wolf + non_wolf_nobel + non_wolf_non_nobel ∧
      -- Consistency check for Nobel laureates
      nobel = wolf_nobel + non_wolf_nobel ∧
      -- The total number of scientists is 50
      total = 50

theorem workshop_attendance_proof : workshop_attendance 50 := by
  sorry

end workshop_attendance_workshop_attendance_proof_l2648_264809


namespace sum_22_probability_l2648_264805

/-- Represents a 20-faced die with some numbered faces and some blank faces -/
structure Die where
  numbered_faces : Finset ℕ
  blank_faces : ℕ
  total_faces : numbered_faces.card + blank_faces = 20

/-- The first die with faces 1 through 18 and two blank faces -/
def die1 : Die where
  numbered_faces := Finset.range 18
  blank_faces := 2
  total_faces := sorry

/-- The second die with faces 2 through 9 and 11 through 20 and two blank faces -/
def die2 : Die where
  numbered_faces := (Finset.range 8).image (λ x => x + 2) ∪ (Finset.range 10).image (λ x => x + 11)
  blank_faces := 2
  total_faces := sorry

/-- The probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- The theorem to be proved -/
theorem sum_22_probability :
  probability (die1.numbered_faces.card * die2.numbered_faces.card) (20 * 20) = 1 / 40 := by
  sorry

end sum_22_probability_l2648_264805


namespace combined_population_theorem_l2648_264835

def wellington_population : ℕ := 900

def port_perry_population (wellington : ℕ) : ℕ := 7 * wellington

def lazy_harbor_population (port_perry : ℕ) : ℕ := port_perry - 800

theorem combined_population_theorem (wellington : ℕ) (port_perry : ℕ) (lazy_harbor : ℕ) :
  wellington = wellington_population →
  port_perry = port_perry_population wellington →
  lazy_harbor = lazy_harbor_population port_perry →
  port_perry + lazy_harbor = 11800 :=
by
  sorry

end combined_population_theorem_l2648_264835


namespace infinite_sum_reciprocal_squared_plus_two_l2648_264855

/-- The infinite sum of 1/(n^2(n+2)) from n=1 to infinity is equal to π^2/12 -/
theorem infinite_sum_reciprocal_squared_plus_two : 
  ∑' (n : ℕ), 1 / (n^2 * (n + 2 : ℝ)) = π^2 / 12 := by sorry

end infinite_sum_reciprocal_squared_plus_two_l2648_264855


namespace negation_of_p_l2648_264819

def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

theorem negation_of_p (f : ℝ → ℝ) :
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
by sorry

end negation_of_p_l2648_264819


namespace equation_solution_l2648_264826

theorem equation_solution : ∃! x : ℝ, 4 * x - 8 + 3 * x = 12 + 5 * x ∧ x = 10 := by
  sorry

end equation_solution_l2648_264826


namespace license_plate_count_l2648_264854

/-- The number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The total number of characters in the license plate -/
def total_chars : ℕ := 8

/-- The number of digits in the license plate -/
def num_plate_digits : ℕ := 6

/-- The number of letters in the license plate -/
def num_plate_letters : ℕ := 2

/-- The number of positions where the two-letter word can be placed -/
def word_positions : ℕ := total_chars - num_plate_letters + 1

/-- The number of positions for the fixed digit 7 -/
def fixed_digit_positions : ℕ := total_chars - 1

theorem license_plate_count :
  (fixed_digit_positions) * (num_letters ^ num_plate_letters) * (num_digits ^ (num_plate_digits - 1)) = 47320000 :=
by sorry

end license_plate_count_l2648_264854


namespace product_xyz_l2648_264886

theorem product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 162)
  (h2 : y * (z + x) = 180)
  (h3 : z * (x + y) = 198)
  (h4 : x + y + z = 26) :
  x * y * z = 2294.67 := by
sorry

end product_xyz_l2648_264886


namespace mikey_leaves_left_l2648_264833

/-- The number of leaves Mikey has left after some blow away -/
def leaves_left (initial : ℕ) (blown_away : ℕ) : ℕ :=
  initial - blown_away

/-- Theorem stating that Mikey has 112 leaves left -/
theorem mikey_leaves_left :
  leaves_left 356 244 = 112 := by
  sorry

end mikey_leaves_left_l2648_264833


namespace new_average_weight_l2648_264804

/-- Given 6 people with an average weight of 154 lbs and a 7th person weighing 133 lbs,
    prove that the new average weight of all 7 people is 151 lbs. -/
theorem new_average_weight 
  (initial_people : Nat) 
  (initial_avg_weight : ℚ) 
  (new_person_weight : ℚ) : 
  initial_people = 6 → 
  initial_avg_weight = 154 → 
  new_person_weight = 133 → 
  ((initial_people : ℚ) * initial_avg_weight + new_person_weight) / (initial_people + 1) = 151 := by
  sorry

#check new_average_weight

end new_average_weight_l2648_264804


namespace intersection_sum_l2648_264846

theorem intersection_sum (m b : ℝ) : 
  (2 * m * 3 + 3 = 9) →  -- First line passes through (3, 9)
  (4 * 3 + b = 9) →      -- Second line passes through (3, 9)
  b + 2 * m = -1 := by
sorry

end intersection_sum_l2648_264846


namespace smallest_root_of_quadratic_l2648_264851

theorem smallest_root_of_quadratic (x : ℝ) :
  (10 * x^2 - 48 * x + 44 = 0) →
  (∀ y : ℝ, 10 * y^2 - 48 * y + 44 = 0 → x ≤ y) →
  x = 1.234 := by
sorry

end smallest_root_of_quadratic_l2648_264851


namespace unique_quadratic_root_l2648_264800

theorem unique_quadratic_root (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → m = 0 ∨ m = -1 := by
  sorry

end unique_quadratic_root_l2648_264800


namespace exam_maximum_marks_l2648_264837

/-- Given an exam where:
  1. The passing mark is 80% of the maximum marks.
  2. A student got 200 marks.
  3. The student failed by 200 marks (i.e., needs 200 more marks to pass).
  Prove that the maximum marks for the exam is 500. -/
theorem exam_maximum_marks :
  ∀ (max_marks : ℕ),
  (max_marks : ℚ) * (80 : ℚ) / (100 : ℚ) = (200 : ℚ) + (200 : ℚ) →
  max_marks = 500 := by
sorry

end exam_maximum_marks_l2648_264837


namespace projectile_max_height_l2648_264869

/-- The height of the projectile as a function of time -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

/-- The time at which the projectile reaches its maximum height -/
def t_max : ℝ := 1

theorem projectile_max_height :
  ∃ (max_height : ℝ), max_height = h t_max ∧ 
  ∀ (t : ℝ), h t ≤ max_height ∧
  max_height = 45 := by
sorry

end projectile_max_height_l2648_264869


namespace oranges_thrown_away_l2648_264818

theorem oranges_thrown_away (initial : ℕ) (added : ℕ) (final : ℕ) :
  initial = 40 →
  added = 7 →
  final = 10 →
  initial - (initial - final + added) = 37 := by
sorry

end oranges_thrown_away_l2648_264818


namespace sum_set_bounds_l2648_264898

theorem sum_set_bounds (A : Finset ℕ) (S : Finset ℕ) :
  A.card = 100 →
  S = Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (A.product A) →
  199 ≤ S.card ∧ S.card ≤ 5050 := by
  sorry

end sum_set_bounds_l2648_264898


namespace quadratic_intersection_intersection_points_l2648_264827

/-- Quadratic function f(x) = x^2 - 6x + 2m - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + 2*m - 1

theorem quadratic_intersection (m : ℝ) :
  (∀ x, f m x ≠ 0) ↔ m > 5 :=
sorry

theorem intersection_points :
  let m : ℝ := -3
  (∃ x, f m x = 0 ∧ (x = -1 ∨ x = 7)) ∧
  (f m 0 = -7) :=
sorry

end quadratic_intersection_intersection_points_l2648_264827


namespace positive_integer_square_minus_five_times_zero_l2648_264848

theorem positive_integer_square_minus_five_times_zero (w : ℕ+) 
  (h : w.val ^ 2 - 5 * w.val = 0) : w.val = 5 := by
  sorry

end positive_integer_square_minus_five_times_zero_l2648_264848


namespace unique_plants_count_l2648_264877

/-- Represents a flower bed -/
structure FlowerBed where
  plants : ℕ

/-- Represents the overlap between two flower beds -/
structure Overlap where
  plants : ℕ

/-- Represents the overlap among three flower beds -/
structure TripleOverlap where
  plants : ℕ

/-- Calculates the total number of unique plants across three overlapping flower beds -/
def totalUniquePlants (a b c : FlowerBed) (ab ac bc : Overlap) (abc : TripleOverlap) : ℕ :=
  a.plants + b.plants + c.plants - ab.plants - ac.plants - bc.plants + abc.plants

/-- Theorem stating that the total number of unique plants across three specific overlapping flower beds is 1320 -/
theorem unique_plants_count :
  let a : FlowerBed := ⟨600⟩
  let b : FlowerBed := ⟨550⟩
  let c : FlowerBed := ⟨400⟩
  let ab : Overlap := ⟨60⟩
  let ac : Overlap := ⟨110⟩
  let bc : Overlap := ⟨90⟩
  let abc : TripleOverlap := ⟨30⟩
  totalUniquePlants a b c ab ac bc abc = 1320 := by
  sorry

end unique_plants_count_l2648_264877


namespace quadrilateral_property_l2648_264843

-- Define a quadrilateral as a tuple of four natural numbers
def Quadrilateral := (ℕ × ℕ × ℕ × ℕ)

-- Define a property that each side divides the sum of the other three
def DivisibilityProperty (q : Quadrilateral) : Prop :=
  let (a, b, c, d) := q
  (a ∣ b + c + d) ∧ (b ∣ a + c + d) ∧ (c ∣ a + b + d) ∧ (d ∣ a + b + c)

-- Define a property that at least two sides are equal
def TwoSidesEqual (q : Quadrilateral) : Prop :=
  let (a, b, c, d) := q
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d

-- The main theorem
theorem quadrilateral_property (q : Quadrilateral) :
  DivisibilityProperty q → TwoSidesEqual q :=
by
  sorry


end quadrilateral_property_l2648_264843


namespace sum_of_roots_quadratic_l2648_264861

theorem sum_of_roots_quadratic (m n : ℝ) : 
  (m^2 + 2*m - 1 = 0) → (n^2 + 2*n - 1 = 0) → (m + n = -2) := by
  sorry

end sum_of_roots_quadratic_l2648_264861


namespace quadratic_equation_roots_l2648_264891

theorem quadratic_equation_roots (a : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*a*x₁ + a^2 - 4 = 0) ∧ 
  (x₂^2 - 2*a*x₂ + a^2 - 4 = 0) := by
sorry

end quadratic_equation_roots_l2648_264891


namespace arithmetic_not_geometric_l2648_264810

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r, ∀ n, a (n + 1) = r * a n

theorem arithmetic_not_geometric (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d ∧ a 1 = 2 →
  ¬(d = 4 ↔ geometric_sequence (λ n => a n)) :=
by sorry

end arithmetic_not_geometric_l2648_264810


namespace variety_show_probability_l2648_264802

/-- The probability of selecting exactly one boy who likes variety shows
    when randomly choosing two boys from a group of five, where two like
    variety shows and three do not. -/
theorem variety_show_probability :
  let total_boys : ℕ := 5
  let boys_like_shows : ℕ := 2
  let boys_dislike_shows : ℕ := 3
  let selected_boys : ℕ := 2
  
  boys_like_shows + boys_dislike_shows = total_boys →
  
  (Nat.choose total_boys selected_boys : ℚ) ≠ 0 →
  
  (Nat.choose boys_like_shows 1 * Nat.choose boys_dislike_shows 1 : ℚ) /
  (Nat.choose total_boys selected_boys : ℚ) = 3 / 5 := by
sorry

end variety_show_probability_l2648_264802


namespace total_hamburgers_made_l2648_264874

def initial_hamburgers : ℝ := 9.0
def additional_hamburgers : ℝ := 3.0

theorem total_hamburgers_made :
  initial_hamburgers + additional_hamburgers = 12.0 := by
  sorry

end total_hamburgers_made_l2648_264874


namespace product_mod_eleven_l2648_264815

theorem product_mod_eleven : (103 * 107) % 11 = 10 := by
  sorry

end product_mod_eleven_l2648_264815


namespace complement_intersection_equals_d_l2648_264887

-- Define the universe
def U : Set Char := {'a', 'b', 'c', 'd', 'e'}

-- Define sets M and N
def M : Set Char := {'a', 'b', 'c'}
def N : Set Char := {'a', 'c', 'e'}

-- State the theorem
theorem complement_intersection_equals_d :
  (U \ M) ∩ (U \ N) = {'d'} := by
  sorry

end complement_intersection_equals_d_l2648_264887


namespace parallel_vectors_t_value_l2648_264883

/-- Given vectors a, b, and c in ℝ², prove that if (a - c) is parallel to (tc + b), then t = -24/17 -/
theorem parallel_vectors_t_value (a b c : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (-3, 4))
  (h2 : b = (-1, 5))
  (h3 : c = (2, 3))
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 - c.1, a.2 - c.2) = k • (t * c.1 + b.1, t * c.2 + b.2)) :
  t = -24/17 := by
  sorry

end parallel_vectors_t_value_l2648_264883


namespace no_rational_solutions_for_positive_k_l2648_264890

theorem no_rational_solutions_for_positive_k : ¬ ∃ (k : ℕ+), ∃ (x : ℚ), k.val * x^2 + 16 * x + k.val = 0 := by
  sorry

end no_rational_solutions_for_positive_k_l2648_264890


namespace power_of_three_difference_l2648_264813

theorem power_of_three_difference : 3^(2+3+4) - (3^2 + 3^3 + 3^4) = 19566 := by
  sorry

end power_of_three_difference_l2648_264813


namespace ratio_of_recurring_decimals_l2648_264853

/-- The value of the repeating decimal 0.848484... -/
def recurring_84 : ℚ := 84 / 99

/-- The value of the repeating decimal 0.212121... -/
def recurring_21 : ℚ := 21 / 99

/-- Theorem stating that the ratio of the two repeating decimals is equal to 4 -/
theorem ratio_of_recurring_decimals : recurring_84 / recurring_21 = 4 := by
  sorry

end ratio_of_recurring_decimals_l2648_264853


namespace quarrel_between_opposite_houses_l2648_264850

/-- Represents a house in the square yard -/
inductive House
| NorthEast
| NorthWest
| SouthEast
| SouthWest

/-- Represents a quarrel between two friends -/
structure Quarrel where
  house1 : House
  house2 : House
  day : Nat

/-- Define what it means for two houses to be neighbors -/
def are_neighbors (h1 h2 : House) : Bool :=
  match h1, h2 with
  | House.NorthEast, House.NorthWest => true
  | House.NorthEast, House.SouthEast => true
  | House.NorthWest, House.SouthWest => true
  | House.SouthEast, House.SouthWest => true
  | House.NorthWest, House.NorthEast => true
  | House.SouthEast, House.NorthEast => true
  | House.SouthWest, House.NorthWest => true
  | House.SouthWest, House.SouthEast => true
  | _, _ => false

/-- The main theorem to prove -/
theorem quarrel_between_opposite_houses 
  (total_friends : Nat) 
  (quarrels : List Quarrel) 
  (h1 : total_friends = 77)
  (h2 : quarrels.length = 365)
  (h3 : ∀ q ∈ quarrels, q.house1 ≠ q.house2)
  (h4 : ∀ h1 h2 : House, are_neighbors h1 h2 → 
    ∃ q ∈ quarrels, (q.house1 = h1 ∧ q.house2 = h2) ∨ (q.house1 = h2 ∧ q.house2 = h1)) :
  ∃ q ∈ quarrels, ¬are_neighbors q.house1 q.house2 := by
sorry

end quarrel_between_opposite_houses_l2648_264850


namespace garden_perimeter_is_800_l2648_264880

/-- The perimeter of a rectangular garden with given length and breadth -/
def garden_perimeter (length breadth : ℝ) : ℝ :=
  2 * (length + breadth)

/-- Theorem: The perimeter of a rectangular garden with length 300 m and breadth 100 m is 800 m -/
theorem garden_perimeter_is_800 :
  garden_perimeter 300 100 = 800 := by
  sorry

end garden_perimeter_is_800_l2648_264880


namespace card_game_combinations_l2648_264894

/-- The number of cards in the deck -/
def deck_size : ℕ := 60

/-- The number of cards in a hand -/
def hand_size : ℕ := 12

/-- The number of distinct unordered hands -/
def num_hands : ℕ := 75287520

theorem card_game_combinations :
  Nat.choose deck_size hand_size = num_hands := by
  sorry

end card_game_combinations_l2648_264894


namespace divisibility_theorem_l2648_264840

theorem divisibility_theorem (K M N : ℤ) (hK : K ≠ 0) (hM : M ≠ 0) (hN : N ≠ 0) (hcoprime : Nat.Coprime K.natAbs M.natAbs) :
  ∃ x : ℤ, ∃ y : ℤ, M * x + N = K * y := by
sorry

end divisibility_theorem_l2648_264840


namespace inequality_proof_l2648_264836

theorem inequality_proof (a : ℝ) : (3 * a - 6) * (2 * a^2 - a^3) ≤ 0 := by
  sorry

end inequality_proof_l2648_264836


namespace derivative_exponential_sine_derivative_rational_function_derivative_logarithm_derivative_polynomial_product_derivative_cosine_l2648_264892

-- Function 1: y = e^(sin x)
theorem derivative_exponential_sine (x : ℝ) :
  deriv (fun x => Real.exp (Real.sin x)) x = Real.exp (Real.sin x) * Real.cos x :=
sorry

-- Function 2: y = (x + 3) / (x + 2)
theorem derivative_rational_function (x : ℝ) :
  deriv (fun x => (x + 3) / (x + 2)) x = -1 / (x + 2)^2 :=
sorry

-- Function 3: y = ln(2x + 3)
theorem derivative_logarithm (x : ℝ) :
  deriv (fun x => Real.log (2 * x + 3)) x = 2 / (2 * x + 3) :=
sorry

-- Function 4: y = (x^2 + 2)(2x - 1)
theorem derivative_polynomial_product (x : ℝ) :
  deriv (fun x => (x^2 + 2) * (2 * x - 1)) x = 6 * x^2 - 2 * x + 4 :=
sorry

-- Function 5: y = cos(2x + π/3)
theorem derivative_cosine (x : ℝ) :
  deriv (fun x => Real.cos (2 * x + Real.pi / 3)) x = -2 * Real.sin (2 * x + Real.pi / 3) :=
sorry

end derivative_exponential_sine_derivative_rational_function_derivative_logarithm_derivative_polynomial_product_derivative_cosine_l2648_264892


namespace linear_function_k_value_l2648_264893

/-- Given that the point (-1, -2) lies on the graph of y = kx - 4 and k ≠ 0, prove that k = -2 -/
theorem linear_function_k_value (k : ℝ) : k ≠ 0 ∧ -2 = k * (-1) - 4 → k = -2 := by
  sorry

end linear_function_k_value_l2648_264893


namespace inequality_preservation_l2648_264876

theorem inequality_preservation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) : 
  a - c < b - c := by
sorry

end inequality_preservation_l2648_264876


namespace train_length_l2648_264812

/-- The length of a train that overtakes a motorbike -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  overtake_time = 18 →
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 180 :=
by
  sorry


end train_length_l2648_264812
