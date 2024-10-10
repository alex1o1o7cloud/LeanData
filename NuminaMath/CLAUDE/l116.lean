import Mathlib

namespace gp_sum_equality_l116_11675

/-- Given two geometric progressions (GPs) where the sum of 3n terms of the first GP
    equals the sum of n terms of the second GP, prove that the first term of the second GP
    equals the sum of the first three terms of the first GP. -/
theorem gp_sum_equality (a b q : ℝ) (n : ℕ) (h_q_ne_one : q ≠ 1) :
  a * (q^(3*n) - 1) / (q - 1) = b * (q^(3*n) - 1) / (q^3 - 1) →
  b = a * (1 + q + q^2) :=
by sorry

end gp_sum_equality_l116_11675


namespace tickets_sold_second_week_l116_11610

/-- The number of tickets sold in the second week of a fair, given the total number of tickets,
    tickets sold in the first week, and tickets left to sell. -/
theorem tickets_sold_second_week
  (total_tickets : ℕ)
  (first_week_sales : ℕ)
  (tickets_left : ℕ)
  (h1 : total_tickets = 90)
  (h2 : first_week_sales = 38)
  (h3 : tickets_left = 35) :
  total_tickets - (first_week_sales + tickets_left) = 17 :=
by sorry

end tickets_sold_second_week_l116_11610


namespace min_voters_for_tall_giraffe_win_l116_11664

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : Nat :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The theorem stating the minimum number of voters required for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win (vs : VotingStructure) 
  (h1 : vs.total_voters = 135)
  (h2 : vs.num_districts = 5)
  (h3 : vs.precincts_per_district = 9)
  (h4 : vs.voters_per_precinct = 3)
  (h5 : vs.total_voters = vs.num_districts * vs.precincts_per_district * vs.voters_per_precinct) :
  min_voters_to_win vs = 30 := by
  sorry

#eval min_voters_to_win ⟨135, 5, 9, 3⟩

end min_voters_for_tall_giraffe_win_l116_11664


namespace sufficient_not_necessary_negation_l116_11696

theorem sufficient_not_necessary_negation (p q : Prop) 
  (h_sufficient : p → q) 
  (h_not_necessary : ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end sufficient_not_necessary_negation_l116_11696


namespace james_out_of_pocket_l116_11648

/-- Calculates the total amount James is out of pocket after his Amazon purchases and returns. -/
def total_out_of_pocket (initial_purchase : ℝ) (returned_tv_cost : ℝ) (returned_bike_cost : ℝ) (toaster_cost : ℝ) : ℝ :=
  let returned_items_value := returned_tv_cost + returned_bike_cost
  let after_returns := initial_purchase - returned_items_value
  let sold_bike_cost := returned_bike_cost * 1.2
  let sold_bike_price := sold_bike_cost * 0.8
  let loss_from_bike_sale := sold_bike_cost - sold_bike_price
  after_returns + loss_from_bike_sale + toaster_cost

/-- Theorem stating that James is out of pocket $2020 given the problem conditions. -/
theorem james_out_of_pocket :
  total_out_of_pocket 3000 700 500 100 = 2020 := by
  sorry

end james_out_of_pocket_l116_11648


namespace sum_of_cubes_zero_l116_11625

theorem sum_of_cubes_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -1) : 
  a^3 + b^3 = 0 := by
sorry

end sum_of_cubes_zero_l116_11625


namespace smallest_n_for_logarithm_sum_l116_11692

theorem smallest_n_for_logarithm_sum : ∃ (n : ℕ), n = 3 ∧ 
  (∀ m : ℕ, m < n → 2^(2^(m+1)) < 512) ∧ 
  2^(2^(n+1)) ≥ 512 := by
  sorry

end smallest_n_for_logarithm_sum_l116_11692


namespace food_budget_fraction_l116_11681

theorem food_budget_fraction (grocery_fraction eating_out_fraction : ℚ) 
  (h1 : grocery_fraction = 0.6)
  (h2 : eating_out_fraction = 0.2) : 
  grocery_fraction + eating_out_fraction = 0.8 := by
  sorry

end food_budget_fraction_l116_11681


namespace M_equals_N_l116_11682

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by sorry

end M_equals_N_l116_11682


namespace polynomial_coefficient_sum_l116_11674

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end polynomial_coefficient_sum_l116_11674


namespace circle_center_radius_sum_l116_11666

/-- Given a circle C with equation 2x^2 + 3y - 25 = -y^2 + 12x + 4,
    where (a,b) is the center and r is the radius,
    prove that a + b + r = 6.744 -/
theorem circle_center_radius_sum (x y a b r : ℝ) : 
  (2 * x^2 + 3 * y - 25 = -y^2 + 12 * x + 4) →
  ((x - a)^2 + (y - b)^2 = r^2) →
  (a + b + r = 6.744) := by
  sorry

end circle_center_radius_sum_l116_11666


namespace max_sum_of_products_l116_11695

theorem max_sum_of_products (f g h j : ℕ) : 
  f ∈ ({4, 5, 9, 10} : Set ℕ) →
  g ∈ ({4, 5, 9, 10} : Set ℕ) →
  h ∈ ({4, 5, 9, 10} : Set ℕ) →
  j ∈ ({4, 5, 9, 10} : Set ℕ) →
  f ≠ g ∧ f ≠ h ∧ f ≠ j ∧ g ≠ h ∧ g ≠ j ∧ h ≠ j →
  f < g →
  f * g + g * h + h * j + f * j ≤ 196 :=
by sorry

end max_sum_of_products_l116_11695


namespace correct_sample_size_l116_11637

/-- Represents the sampling strategy for a company's employee health survey. -/
structure CompanySampling where
  total_employees : ℕ
  young_employees : ℕ
  middle_aged_employees : ℕ
  elderly_employees : ℕ
  young_in_sample : ℕ

/-- The sample size for the company's health survey. -/
def sample_size (cs : CompanySampling) : ℕ := 15

theorem correct_sample_size (cs : CompanySampling) 
  (h1 : cs.total_employees = 750)
  (h2 : cs.young_employees = 350)
  (h3 : cs.middle_aged_employees = 250)
  (h4 : cs.elderly_employees = 150)
  (h5 : cs.young_in_sample = 7) :
  sample_size cs = 15 := by
  sorry

#check correct_sample_size

end correct_sample_size_l116_11637


namespace sprocket_production_rate_l116_11656

/-- The number of sprockets both machines produce -/
def total_sprockets : ℕ := 330

/-- The additional time (in hours) machine A takes compared to machine B -/
def time_difference : ℕ := 10

/-- The production rate increase of machine B compared to machine A -/
def rate_increase : ℚ := 1/10

/-- The production rate of machine A in sprockets per hour -/
def machine_a_rate : ℚ := 3

/-- The production rate of machine B in sprockets per hour -/
def machine_b_rate : ℚ := machine_a_rate * (1 + rate_increase)

/-- The time taken by machine A to produce the total sprockets -/
def machine_a_time : ℚ := total_sprockets / machine_a_rate

/-- The time taken by machine B to produce the total sprockets -/
def machine_b_time : ℚ := total_sprockets / machine_b_rate

theorem sprocket_production_rate :
  (machine_a_time = machine_b_time + time_difference) ∧
  (machine_b_rate = machine_a_rate * (1 + rate_increase)) ∧
  (total_sprockets = machine_a_rate * machine_a_time) ∧
  (total_sprockets = machine_b_rate * machine_b_time) :=
sorry

end sprocket_production_rate_l116_11656


namespace fraction_simplification_l116_11647

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end fraction_simplification_l116_11647


namespace twenty_fifth_triangular_number_l116_11614

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 25th triangular number is 325 -/
theorem twenty_fifth_triangular_number : triangular_number 25 = 325 := by
  sorry

end twenty_fifth_triangular_number_l116_11614


namespace equation_root_existence_l116_11661

theorem equation_root_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ a + b ∧ x₀ = a * Real.sin x₀ + b := by
  sorry

end equation_root_existence_l116_11661


namespace negation_of_forall_gt_is_exists_leq_l116_11644

theorem negation_of_forall_gt_is_exists_leq :
  (¬ ∀ x : ℝ, x^2 > 1 - 2*x) ↔ (∃ x : ℝ, x^2 ≤ 1 - 2*x) := by sorry

end negation_of_forall_gt_is_exists_leq_l116_11644


namespace airport_distance_proof_l116_11622

/-- The distance from David's home to the airport in miles -/
def airport_distance : ℝ := 155

/-- David's initial speed in miles per hour -/
def initial_speed : ℝ := 45

/-- The increase in speed for the remaining journey in miles per hour -/
def speed_increase : ℝ := 20

/-- The time David would be late if he continued at the initial speed, in hours -/
def late_time : ℝ := 0.75

theorem airport_distance_proof :
  ∃ (t : ℝ),
    -- t is the actual time needed to arrive on time
    t > 0 ∧
    -- The total distance equals the distance covered at the initial speed
    airport_distance = initial_speed * (t + late_time) ∧
    -- The remaining distance equals the distance covered at the increased speed
    airport_distance - initial_speed = (initial_speed + speed_increase) * (t - 1) :=
by sorry

#check airport_distance_proof

end airport_distance_proof_l116_11622


namespace mike_marbles_l116_11632

/-- Calculates the number of marbles Mike has after giving some to Sam. -/
def marblesLeft (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Proves that Mike has 4 marbles left after giving 4 out of his initial 8 marbles to Sam. -/
theorem mike_marbles : marblesLeft 8 4 = 4 := by
  sorry

end mike_marbles_l116_11632


namespace age_ratio_correct_l116_11639

-- Define Sachin's age
def sachin_age : ℚ := 24.5

-- Define the age difference between Rahul and Sachin
def age_difference : ℚ := 7

-- Calculate Rahul's age
def rahul_age : ℚ := sachin_age + age_difference

-- Define the ratio of their ages
def age_ratio : ℚ × ℚ := (7, 9)

-- Theorem to prove
theorem age_ratio_correct : 
  (sachin_age / rahul_age) = (age_ratio.1 / age_ratio.2) := by
  sorry

end age_ratio_correct_l116_11639


namespace jennas_profit_l116_11685

/-- Calculates the profit for Jenna's wholesale business --/
def calculate_profit (
  widget_cost : ℝ)
  (widget_price : ℝ)
  (rent : ℝ)
  (tax_rate : ℝ)
  (worker_salary : ℝ)
  (num_workers : ℕ)
  (widgets_sold : ℕ) : ℝ :=
  let revenue := widget_price * widgets_sold
  let cost_of_goods_sold := widget_cost * widgets_sold
  let gross_profit := revenue - cost_of_goods_sold
  let fixed_costs := rent + (worker_salary * num_workers)
  let profit_before_tax := gross_profit - fixed_costs
  let tax := tax_rate * profit_before_tax
  profit_before_tax - tax

/-- Theorem stating that Jenna's profit is $4000 given the specified conditions --/
theorem jennas_profit :
  calculate_profit 3 8 10000 0.2 2500 4 5000 = 4000 := by
  sorry

end jennas_profit_l116_11685


namespace correct_calculation_l116_11601

theorem correct_calculation (x : ℝ) : 14 * x = 70 → x - 6 = -1 := by
  sorry

end correct_calculation_l116_11601


namespace factorial_ratio_l116_11659

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 10 = 132 := by
  sorry

end factorial_ratio_l116_11659


namespace final_number_in_range_l116_11662

def A : List Nat := List.range (2016 - 672 + 1) |>.map (· + 672)

def replace_step (numbers : List Rat) : List Rat :=
  let (a, b, c) := (numbers.get! 0, numbers.get! 1, numbers.get! 2)
  let new_num := (1 : Rat) / 3 * min a (min b c)
  new_num :: numbers.drop 3

def iterate_replacement (numbers : List Rat) (n : Nat) : List Rat :=
  match n with
  | 0 => numbers
  | n + 1 => iterate_replacement (replace_step numbers) n

theorem final_number_in_range :
  let initial_numbers := A.map (λ x => (x : Rat))
  let final_list := iterate_replacement initial_numbers 672
  final_list.length = 1 ∧ 0 < final_list.head! ∧ final_list.head! < 1 := by
  sorry

end final_number_in_range_l116_11662


namespace fraction_equality_l116_11616

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := by
  sorry

end fraction_equality_l116_11616


namespace sin_cos_sum_27_18_l116_11658

theorem sin_cos_sum_27_18 :
  Real.sin (27 * π / 180) * Real.cos (18 * π / 180) +
  Real.cos (27 * π / 180) * Real.sin (18 * π / 180) =
  Real.sqrt 2 / 2 :=
by sorry

end sin_cos_sum_27_18_l116_11658


namespace min_stamps_for_37_cents_l116_11617

/-- Represents the number of ways to make a certain amount with given coin denominations -/
def numWays (amount : ℕ) (coins : List ℕ) : ℕ := sorry

/-- Finds the minimum number of coins needed to make the amount -/
def minCoins (amount : ℕ) (coins : List ℕ) : ℕ := sorry

theorem min_stamps_for_37_cents :
  minCoins 37 [5, 7] = 7 := by sorry

end min_stamps_for_37_cents_l116_11617


namespace literate_female_percentage_approx_81_percent_l116_11608

/-- Represents the demographics and literacy rates of a town -/
structure TownDemographics where
  total_inhabitants : ℕ
  adult_male_percent : ℚ
  adult_female_percent : ℚ
  children_percent : ℚ
  adult_male_literacy : ℚ
  adult_female_literacy : ℚ
  children_literacy : ℚ

/-- Calculates the percentage of literate females in the town -/
def literate_female_percentage (town : TownDemographics) : ℚ :=
  let adult_females := town.total_inhabitants * town.adult_female_percent
  let female_children := town.total_inhabitants * town.children_percent / 2
  let literate_adult_females := adult_females * town.adult_female_literacy
  let literate_female_children := female_children * town.children_literacy
  let total_literate_females := literate_adult_females + literate_female_children
  let total_females := adult_females + female_children
  total_literate_females / total_females

/-- Theorem stating that the percentage of literate females in the town is approximately 81% -/
theorem literate_female_percentage_approx_81_percent 
  (town : TownDemographics)
  (h1 : town.total_inhabitants = 3500)
  (h2 : town.adult_male_percent = 60 / 100)
  (h3 : town.adult_female_percent = 35 / 100)
  (h4 : town.children_percent = 5 / 100)
  (h5 : town.adult_male_literacy = 55 / 100)
  (h6 : town.adult_female_literacy = 80 / 100)
  (h7 : town.children_literacy = 95 / 100) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 / 100 ∧ 
  |literate_female_percentage town - 81 / 100| < ε :=
sorry

end literate_female_percentage_approx_81_percent_l116_11608


namespace sum_equality_l116_11634

theorem sum_equality : 9548 + 7314 = 3362 + 13500 := by
  sorry

end sum_equality_l116_11634


namespace smallest_stamp_collection_l116_11660

theorem smallest_stamp_collection (M : ℕ) : 
  M > 2 →
  M % 5 = 2 →
  M % 7 = 2 →
  M % 9 = 2 →
  (∀ N : ℕ, N > 2 ∧ N % 5 = 2 ∧ N % 7 = 2 ∧ N % 9 = 2 → N ≥ M) →
  M = 317 :=
by sorry

end smallest_stamp_collection_l116_11660


namespace basketball_free_throws_l116_11684

theorem basketball_free_throws (two_pointers three_pointers free_throws : ℕ) : 
  (3 * three_pointers = 2 * two_pointers) →
  (free_throws = three_pointers) →
  (2 * two_pointers + 3 * three_pointers + free_throws = 73) →
  free_throws = 10 := by
sorry

end basketball_free_throws_l116_11684


namespace intersection_M_N_l116_11649

def M : Set ℝ := {x | x > -3}
def N : Set ℝ := {x | x ≥ 2}

theorem intersection_M_N : M ∩ N = Set.Ici 2 := by sorry

end intersection_M_N_l116_11649


namespace problem_solution_l116_11609

theorem problem_solution (x : ℝ) (h1 : x < 0) (h2 : 1 / (x + 1 / (x + 2)) = 2) : x + 7/2 = 2 := by
  sorry

end problem_solution_l116_11609


namespace handshakes_in_gathering_l116_11686

/-- The number of handshakes in a gathering with specific conditions -/
def number_of_handshakes (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: In a gathering of 8 married couples with specific handshake rules, there are 104 handshakes -/
theorem handshakes_in_gathering : number_of_handshakes 16 = 104 := by
  sorry

end handshakes_in_gathering_l116_11686


namespace quadratic_properties_l116_11626

/-- The quadratic function f(x) = x^2 - 4x - 5 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 5

theorem quadratic_properties :
  (∀ x, f x ≥ -9) ∧ 
  (f 5 = 0 ∧ f (-1) = 0) :=
sorry

end quadratic_properties_l116_11626


namespace petes_flag_has_128_shapes_l116_11621

/-- Calculates the total number of shapes on Pete's flag given the number of stars and stripes on the US flag. -/
def petes_flag_shapes (us_stars : ℕ) (us_stripes : ℕ) : ℕ :=
  let circles := us_stars / 2 - 3
  let squares := 2 * us_stripes + 6
  let triangles := 2 * (us_stars - us_stripes)
  circles + squares + triangles

/-- Theorem stating that Pete's flag has 128 shapes given the US flag has 50 stars and 13 stripes. -/
theorem petes_flag_has_128_shapes :
  petes_flag_shapes 50 13 = 128 := by
  sorry

end petes_flag_has_128_shapes_l116_11621


namespace fraction_pattern_l116_11641

theorem fraction_pattern (n m k : ℕ) (h1 : m ≠ 0) (h2 : k ≠ 0) 
  (h3 : n / m = k * n / (k * m)) : 
  (n + m) / m = (k * n + k * m) / (k * m) := by
  sorry

end fraction_pattern_l116_11641


namespace inscribed_circle_area_l116_11643

theorem inscribed_circle_area (large_square_area : ℝ) (h : large_square_area = 80) :
  let large_side := Real.sqrt large_square_area
  let small_side := large_side / Real.sqrt 2
  let circle_radius := small_side / 2
  circle_radius ^ 2 * Real.pi = 10 * Real.pi :=
by sorry

end inscribed_circle_area_l116_11643


namespace disobedient_pair_implies_ultra_disobedient_l116_11624

/-- A function from natural numbers to positive real numbers -/
def IncreasingPositiveFunction : Type := 
  {f : ℕ → ℝ // (∀ m n, m < n → f m < f n) ∧ (∀ n, f n > 0)}

/-- Definition of a disobedient pair -/
def IsDisobedientPair (f : IncreasingPositiveFunction) (m n : ℕ) : Prop :=
  f.val (m * n) ≠ f.val m * f.val n

/-- Definition of an ultra-disobedient number -/
def IsUltraDisobedient (f : IncreasingPositiveFunction) (m : ℕ) : Prop :=
  ∀ N : ℕ, ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 
    ∀ i : ℕ, i ≤ N → IsDisobedientPair f m (n + i)

/-- Main theorem: existence of a disobedient pair implies existence of an ultra-disobedient number -/
theorem disobedient_pair_implies_ultra_disobedient
  (f : IncreasingPositiveFunction)
  (h : ∃ m n : ℕ, IsDisobedientPair f m n) :
  ∃ m : ℕ, IsUltraDisobedient f m :=
sorry

end disobedient_pair_implies_ultra_disobedient_l116_11624


namespace scores_statistics_l116_11619

def scores : List ℕ := [98, 88, 90, 92, 90, 94]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def average (l : List ℕ) : ℚ := sorry

theorem scores_statistics :
  mode scores = 90 ∧
  median scores = 91 ∧
  average scores = 92 := by sorry

end scores_statistics_l116_11619


namespace room_width_proof_l116_11652

theorem room_width_proof (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 5.5 → 
  cost_per_sqm = 750 → 
  total_cost = 16500 → 
  width * length * cost_per_sqm = total_cost → 
  width = 4 := by
sorry

end room_width_proof_l116_11652


namespace bob_picked_450_apples_l116_11683

/-- The number of apples Bob picked for his family -/
def apples_picked (num_children : ℕ) (apples_per_child : ℕ) (num_adults : ℕ) (apples_per_adult : ℕ) : ℕ :=
  num_children * apples_per_child + num_adults * apples_per_adult

/-- Theorem stating that Bob picked 450 apples for his family -/
theorem bob_picked_450_apples : 
  apples_picked 33 10 40 3 = 450 := by
  sorry

end bob_picked_450_apples_l116_11683


namespace pentagon_largest_angle_l116_11680

/-- 
Given a convex pentagon with interior angles measuring y, 2y+2, 3y-3, 4y+4, and 5y-5 degrees,
where the sum of these angles is 540 degrees, prove that the largest angle measures 176 degrees
when rounded to the nearest integer.
-/
theorem pentagon_largest_angle : 
  ∀ y : ℝ, 
  y + (2*y+2) + (3*y-3) + (4*y+4) + (5*y-5) = 540 → 
  round (5*y - 5) = 176 := by
sorry

end pentagon_largest_angle_l116_11680


namespace sum_of_three_numbers_l116_11679

theorem sum_of_three_numbers (second : ℕ) (h1 : second = 30) : ∃ (first third : ℕ),
  first = 2 * second ∧ 
  third = first / 3 ∧ 
  first + second + third = 110 := by
sorry

end sum_of_three_numbers_l116_11679


namespace figure_area_solution_l116_11631

theorem figure_area_solution (x : ℝ) : 
  let square1_area := (3*x)^2
  let square2_area := (7*x)^2
  let triangle_area := (1/2) * (3*x) * (7*x)
  let total_area := square1_area + square2_area + triangle_area
  total_area = 1360 → x = Real.sqrt (2720/119) := by
sorry

end figure_area_solution_l116_11631


namespace storage_unit_solution_l116_11611

/-- Represents the storage unit problem -/
def storage_unit_problem (total_units : ℕ) (small_units : ℕ) (small_length : ℕ) (small_width : ℕ) (large_area : ℕ) : Prop :=
  let small_area : ℕ := small_length * small_width
  let large_units : ℕ := total_units - small_units
  let total_area : ℕ := small_units * small_area + large_units * large_area
  total_area = 5040

/-- Theorem stating the solution to the storage unit problem -/
theorem storage_unit_solution : storage_unit_problem 42 20 8 4 200 := by
  sorry


end storage_unit_solution_l116_11611


namespace simplify_fraction_product_l116_11673

theorem simplify_fraction_product : 18 * (8 / 15) * (2 / 27) = 32 / 45 := by
  sorry

end simplify_fraction_product_l116_11673


namespace singer_arrangements_l116_11646

/-- The number of ways to arrange n objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k objects out of n objects. -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem singer_arrangements : 
  let total_singers : ℕ := 5
  let arrangements_case1 := permutations 4 4  -- when the singer who can't be last is first
  let arrangements_case2 := permutations 3 1 * permutations 3 1 * permutations 3 3  -- other cases
  arrangements_case1 + arrangements_case2 = 78 := by
  sorry

end singer_arrangements_l116_11646


namespace sum_of_specific_polynomials_l116_11642

/-- A linear polynomial -/
def LinearPolynomial (α : Type*) [Field α] := α → α

/-- A cubic polynomial -/
def CubicPolynomial (α : Type*) [Field α] := α → α

/-- The theorem statement -/
theorem sum_of_specific_polynomials 
  (p : LinearPolynomial ℝ) (q : CubicPolynomial ℝ)
  (h1 : p 1 = 1)
  (h2 : q (-1) = -3)
  (h3 : ∃ r : ℝ → ℝ, ∀ x, q x = r x * (x - 2)^2)
  (h4 : ∃ s t : ℝ → ℝ, (∀ x, p x = s x * (x + 1)) ∧ (∀ x, q x = t x * (x + 1)))
  : ∀ x, p x + q x = -1/3 * x^3 + 4/3 * x^2 + 1/3 * x + 13/6 := by
  sorry

end sum_of_specific_polynomials_l116_11642


namespace uniform_motion_parametric_equation_l116_11603

/-- Parametric equation of a point undergoing uniform linear motion -/
def parametric_equation (initial_x initial_y vx vy : ℝ) : ℝ → ℝ × ℝ :=
  λ t => (initial_x + vx * t, initial_y + vy * t)

/-- The correct parametric equation for the given conditions -/
theorem uniform_motion_parametric_equation :
  parametric_equation 1 1 9 12 = λ t => (1 + 9 * t, 1 + 12 * t) := by
  sorry

end uniform_motion_parametric_equation_l116_11603


namespace fraction_equality_implies_equality_l116_11651

theorem fraction_equality_implies_equality (x y : ℝ) : x / 2 = y / 2 → x = y := by
  sorry

end fraction_equality_implies_equality_l116_11651


namespace equal_cost_at_150_miles_unique_equal_cost_mileage_l116_11602

-- Define the cost functions for both rental companies
def safety_cost (m : ℝ) : ℝ := 41.95 + 0.29 * m
def city_cost (m : ℝ) : ℝ := 38.95 + 0.31 * m

-- Theorem stating that the costs are equal at 150 miles
theorem equal_cost_at_150_miles : 
  safety_cost 150 = city_cost 150 := by
  sorry

-- Theorem stating that 150 miles is the unique solution
theorem unique_equal_cost_mileage :
  ∀ m : ℝ, safety_cost m = city_cost m ↔ m = 150 := by
  sorry

end equal_cost_at_150_miles_unique_equal_cost_mileage_l116_11602


namespace median_of_four_numbers_l116_11613

theorem median_of_four_numbers (x : ℝ) : 
  (0 < 4) ∧ (4 < x) ∧ (x < 10) ∧  -- ascending order condition
  ((4 + x) / 2 = 5)                -- median condition
  → x = 6 := by
sorry

end median_of_four_numbers_l116_11613


namespace first_floor_bedrooms_count_l116_11665

/-- Represents a two-story house with bedrooms -/
structure House where
  total_bedrooms : ℕ
  second_floor_bedrooms : ℕ

/-- Calculates the number of bedrooms on the first floor -/
def first_floor_bedrooms (h : House) : ℕ :=
  h.total_bedrooms - h.second_floor_bedrooms

/-- Theorem: For a house with 10 total bedrooms and 2 bedrooms on the second floor,
    the first floor has 8 bedrooms -/
theorem first_floor_bedrooms_count (h : House) 
    (h_total : h.total_bedrooms = 10)
    (h_second : h.second_floor_bedrooms = 2) : 
    first_floor_bedrooms h = 8 := by
  sorry

end first_floor_bedrooms_count_l116_11665


namespace average_licks_to_center_l116_11697

def dan_licks : ℕ := 58
def michael_licks : ℕ := 63
def sam_licks : ℕ := 70
def david_licks : ℕ := 70
def lance_licks : ℕ := 39

def total_licks : ℕ := dan_licks + michael_licks + sam_licks + david_licks + lance_licks
def num_people : ℕ := 5

theorem average_licks_to_center (h : total_licks = dan_licks + michael_licks + sam_licks + david_licks + lance_licks) :
  (total_licks : ℚ) / num_people = 60 := by sorry

end average_licks_to_center_l116_11697


namespace evaluate_sqrt_fraction_l116_11671

theorem evaluate_sqrt_fraction (y : ℝ) (h : y < -2) :
  Real.sqrt (y / (1 - (y + 1) / (y + 2))) = -y := by
  sorry

end evaluate_sqrt_fraction_l116_11671


namespace meet_once_l116_11691

/-- Represents the meeting scenario between Michael and the garbage truck --/
structure MeetingScenario where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once (scenario : MeetingScenario) : 
  scenario.michael_speed = 6 ∧ 
  scenario.truck_speed = 10 ∧ 
  scenario.pail_distance = 200 ∧ 
  scenario.truck_stop_time = 30 ∧
  scenario.initial_distance = 200 →
  number_of_meetings scenario = 1 :=
sorry

end meet_once_l116_11691


namespace comparison_theorem_l116_11688

theorem comparison_theorem :
  let a : ℝ := (5/3)^(1/5)
  let b : ℝ := (2/3)^10
  let c : ℝ := Real.log 6 / Real.log 0.3
  a > b ∧ b > c := by sorry

end comparison_theorem_l116_11688


namespace total_cost_is_correct_l116_11623

-- Define the cost of one t-shirt
def cost_per_shirt : ℚ := 9.95

-- Define the number of t-shirts bought
def num_shirts : ℕ := 25

-- Define the total cost
def total_cost : ℚ := cost_per_shirt * num_shirts

-- Theorem to prove
theorem total_cost_is_correct : total_cost = 248.75 := by
  sorry

end total_cost_is_correct_l116_11623


namespace sum_divisible_by_31_l116_11605

def geometric_sum (n : ℕ) : ℕ := (2^(5*n) - 1) / (2 - 1)

theorem sum_divisible_by_31 (n : ℕ+) : 
  31 ∣ geometric_sum n.val := by sorry

end sum_divisible_by_31_l116_11605


namespace better_fit_model_l116_11669

def sum_of_squared_residuals (model : Nat) : ℝ :=
  if model = 1 then 153.4 else 200

def better_fit (model1 model2 : Nat) : Prop :=
  sum_of_squared_residuals model1 < sum_of_squared_residuals model2

theorem better_fit_model : better_fit 1 2 :=
by sorry

end better_fit_model_l116_11669


namespace leakage_time_to_empty_tank_l116_11699

/-- Given a pipe that takes 'a' hours to fill a tank without leakage,
    and 7a hours to fill the tank with leakage, prove that the time 'l'
    taken by the leakage alone to empty the tank is equal to 7a/6 hours. -/
theorem leakage_time_to_empty_tank (a : ℝ) (h : a > 0) :
  let l : ℝ := (7 * a) / 6
  let fill_rate : ℝ := 1 / a
  let leak_rate : ℝ := 1 / l
  fill_rate - leak_rate = 1 / (7 * a) :=
by sorry

end leakage_time_to_empty_tank_l116_11699


namespace zero_full_crates_l116_11636

/-- Represents the number of berries picked for each type -/
structure BerriesPicked where
  blueberries : ℕ
  cranberries : ℕ
  raspberries : ℕ
  gooseberries : ℕ
  strawberries : ℕ

/-- Represents the fraction of rotten berries for each type -/
structure RottenFractions where
  blueberries : ℚ
  cranberries : ℚ
  raspberries : ℚ
  gooseberries : ℚ
  strawberries : ℚ

/-- Represents the number of berries required to fill one crate for each type -/
structure CrateCapacity where
  blueberries : ℕ
  cranberries : ℕ
  raspberries : ℕ
  gooseberries : ℕ
  strawberries : ℕ

/-- Calculates the number of full crates that can be sold -/
def calculateFullCrates (picked : BerriesPicked) (rotten : RottenFractions) (capacity : CrateCapacity) : ℕ :=
  sorry

/-- Theorem stating that the number of full crates that can be sold is 0 -/
theorem zero_full_crates : 
  let picked : BerriesPicked := ⟨30, 20, 10, 15, 25⟩
  let rotten : RottenFractions := ⟨1/3, 1/4, 1/5, 1/6, 1/7⟩
  let capacity : CrateCapacity := ⟨40, 50, 30, 60, 70⟩
  calculateFullCrates picked rotten capacity = 0 := by
  sorry

end zero_full_crates_l116_11636


namespace town_population_theorem_l116_11676

theorem town_population_theorem (total_population : ℕ) (num_groups : ℕ) (male_groups : ℕ) :
  total_population = 450 →
  num_groups = 4 →
  male_groups = 2 →
  (male_groups * (total_population / num_groups) : ℕ) = 225 :=
by sorry

end town_population_theorem_l116_11676


namespace quadratic_coefficient_l116_11689

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- Theorem: For a quadratic function with integer coefficients, 
    if its vertex is at (2, 5) and it passes through (1, 4), 
    then its leading coefficient is -1 -/
theorem quadratic_coefficient (q : QuadraticFunction) 
  (vertex : q.f 2 = 5) 
  (point : q.f 1 = 4) : 
  q.a = -1 := by sorry

end quadratic_coefficient_l116_11689


namespace f_properties_l116_11670

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a| + |2*x - 1/a|

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x, f 1 x ≤ 6 ↔ x ∈ Set.Icc (-7/3) (5/3)) ∧
  (∀ x, f a x ≥ 2) :=
sorry

end f_properties_l116_11670


namespace sin_cos_sixth_power_sum_l116_11600

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 5) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 97 / 100 := by
  sorry

end sin_cos_sixth_power_sum_l116_11600


namespace average_children_in_families_with_children_l116_11628

/-- Given 15 families with an average of 3 children per family, 
    and exactly 3 of these families being childless, 
    prove that the average number of children in the families 
    that have children is 45/12. -/
theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children = 3)
  (h3 : childless_families = 3) :
  (total_families : ℚ) * average_children / 
  ((total_families : ℚ) - childless_families) = 45 / 12 := by
sorry

end average_children_in_families_with_children_l116_11628


namespace buddy_fraction_l116_11672

theorem buddy_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  (s : ℚ) / 3 = (n : ℚ) / 4 →
  ((s : ℚ) / 3 + (n : ℚ) / 4) / ((s : ℚ) + (n : ℚ)) = 2 / 7 := by
  sorry

end buddy_fraction_l116_11672


namespace hexagon_central_symmetry_l116_11698

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → Point

/-- Checks if a hexagon is centrally symmetric -/
def isCentrallySymmetric (h : Hexagon) : Prop := sorry

/-- Checks if a hexagon is regular -/
def isRegular (h : Hexagon) : Prop := sorry

/-- Constructs equilateral triangles on each side of the hexagon -/
def constructOutwardTriangles (h : Hexagon) : Fin 6 → EquilateralTriangle := sorry

/-- Finds the midpoints of the sides of the new hexagon formed by the triangle vertices -/
def findMidpoints (h : Hexagon) (triangles : Fin 6 → EquilateralTriangle) : Hexagon := sorry

/-- The main theorem -/
theorem hexagon_central_symmetry 
  (h : Hexagon) 
  (triangles : Fin 6 → EquilateralTriangle)
  (midpoints : Hexagon) 
  (h_triangles : triangles = constructOutwardTriangles h)
  (h_midpoints : midpoints = findMidpoints h triangles)
  (h_regular : isRegular midpoints) :
  isCentrallySymmetric h := sorry

end hexagon_central_symmetry_l116_11698


namespace enclosing_rectangle_exists_l116_11615

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  is_convex : Convex ℝ vertices
  area : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  bottom_left : ℝ × ℝ
  top_right : ℝ × ℝ

/-- Checks if a polygon is enclosed within a rectangle -/
def enclosed (p : ConvexPolygon) (r : Rectangle) : Prop :=
  ∀ v ∈ p.vertices, 
    r.bottom_left.1 ≤ v.1 ∧ v.1 ≤ r.top_right.1 ∧
    r.bottom_left.2 ≤ v.2 ∧ v.2 ≤ r.top_right.2

/-- Calculates the area of a rectangle -/
def rectangle_area (r : Rectangle) : ℝ :=
  (r.top_right.1 - r.bottom_left.1) * (r.top_right.2 - r.bottom_left.2)

/-- The main theorem -/
theorem enclosing_rectangle_exists (p : ConvexPolygon) (h : p.area = 1) :
  ∃ r : Rectangle, enclosed p r ∧ rectangle_area r ≤ 2 := by
  sorry

end enclosing_rectangle_exists_l116_11615


namespace solution_set_solves_inequality_l116_11655

/-- The solution set of the inequality 12x^2 - ax > a^2 for a given real number a -/
def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then {x | x < -a/4 ∨ x > a/3}
  else if a = 0 then {x | x ≠ 0}
  else {x | x < a/3 ∨ x > -a/4}

/-- Theorem stating that the solution_set function correctly solves the inequality -/
theorem solution_set_solves_inequality (a : ℝ) :
  ∀ x, x ∈ solution_set a ↔ 12 * x^2 - a * x > a^2 :=
by sorry

end solution_set_solves_inequality_l116_11655


namespace jodi_walked_3_miles_week3_l116_11629

/-- Represents the walking schedule of Jodi over 4 weeks -/
structure WalkingSchedule where
  weeks : Nat
  days_per_week : Nat
  miles_week1 : Nat
  miles_week2 : Nat
  miles_week4 : Nat
  total_miles : Nat

/-- Calculates the miles walked per day in the third week -/
def miles_per_day_week3 (schedule : WalkingSchedule) : Nat :=
  let miles_weeks_124 := schedule.miles_week1 * schedule.days_per_week +
                         schedule.miles_week2 * schedule.days_per_week +
                         schedule.miles_week4 * schedule.days_per_week
  let miles_week3 := schedule.total_miles - miles_weeks_124
  miles_week3 / schedule.days_per_week

/-- Theorem stating that Jodi walked 3 miles per day in the third week -/
theorem jodi_walked_3_miles_week3 (schedule : WalkingSchedule) 
  (h1 : schedule.weeks = 4)
  (h2 : schedule.days_per_week = 6)
  (h3 : schedule.miles_week1 = 1)
  (h4 : schedule.miles_week2 = 2)
  (h5 : schedule.miles_week4 = 4)
  (h6 : schedule.total_miles = 60) :
  miles_per_day_week3 schedule = 3 := by
  sorry

end jodi_walked_3_miles_week3_l116_11629


namespace total_commute_time_is_16_l116_11612

/-- Time it takes Roque to walk to work (in hours) -/
def walk_time : ℕ := 2

/-- Time it takes Roque to bike to work (in hours) -/
def bike_time : ℕ := 1

/-- Number of times Roque walks to and from work per week -/
def walk_frequency : ℕ := 3

/-- Number of times Roque bikes to and from work per week -/
def bike_frequency : ℕ := 2

/-- Total time Roque spends commuting in a week -/
def total_commute_time : ℕ := (walk_time * walk_frequency * 2) + (bike_time * bike_frequency * 2)

theorem total_commute_time_is_16 : total_commute_time = 16 := by
  sorry

end total_commute_time_is_16_l116_11612


namespace line_symmetry_l116_11693

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are symmetric with respect to a third line -/
def symmetric (l1 l2 ls : Line) : Prop :=
  ∀ x y : ℝ, l1.contains x y → 
    ∃ x' y' : ℝ, l2.contains x' y' ∧
      (x + x') / 2 = (y + y') / 2 ∧ ls.contains ((x + x') / 2) ((y + y') / 2)

theorem line_symmetry :
  let l1 : Line := ⟨-2, 1, 1⟩  -- y = 2x + 1
  let l2 : Line := ⟨1, -2, 0⟩  -- x - 2y = 0
  let ls : Line := ⟨1, 1, 1⟩  -- x + y + 1 = 0
  symmetric l1 l2 ls := by sorry

end line_symmetry_l116_11693


namespace jane_quiz_score_l116_11607

/-- Represents the scoring system for a quiz -/
structure QuizScoring where
  correct : Int
  incorrect : Int
  unanswered : Int

/-- Represents a student's quiz results -/
structure QuizResults where
  total : Nat
  correct : Nat
  incorrect : Nat
  unanswered : Nat

/-- Calculates the final score based on quiz results and scoring system -/
def calculateScore (results : QuizResults) (scoring : QuizScoring) : Int :=
  results.correct * scoring.correct + 
  results.incorrect * scoring.incorrect + 
  results.unanswered * scoring.unanswered

/-- Theorem: Jane's final score in the quiz is 20 -/
theorem jane_quiz_score : 
  let scoring : QuizScoring := ⟨2, -1, 0⟩
  let results : QuizResults := ⟨30, 15, 10, 5⟩
  calculateScore results scoring = 20 := by
  sorry


end jane_quiz_score_l116_11607


namespace two_from_ten_for_different_positions_l116_11618

/-- The number of ways to choose k items from n items where order matters -/
def permutations (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

/-- The number of ways to choose 2 people from 10 for 2 different positions -/
theorem two_from_ten_for_different_positions : permutations 10 2 = 90 := by
  sorry

end two_from_ten_for_different_positions_l116_11618


namespace angle_measures_l116_11653

/-- Given supplementary angles A and B, where A is 6 times B, and B forms a complementary angle C,
    prove the measures of angles A, B, and C. -/
theorem angle_measures (A B C : ℝ) : 
  A + B = 180 →  -- A and B are supplementary
  A = 6 * B →    -- A is 6 times B
  B + C = 90 →   -- B and C are complementary
  A = 180 * 6 / 7 ∧ B = 180 / 7 ∧ C = 90 - 180 / 7 := by
  sorry

end angle_measures_l116_11653


namespace divisor_prime_ratio_l116_11657

def d (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_prime_ratio (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  n / d n = p ↔ 
    n = 8 ∨ n = 9 ∨ n = 12 ∨ n = 18 ∨ n = 24 ∨
    (∃ q : ℕ, Nat.Prime q ∧ q > 3 ∧ (n = 8 * q ∨ n = 12 * q)) :=
by sorry

end divisor_prime_ratio_l116_11657


namespace green_tea_price_decrease_proof_l116_11630

/-- The percentage decrease in green tea price from June to July -/
def green_tea_price_decrease : ℝ := 90

/-- The cost per pound of green tea and coffee in June -/
def june_price : ℝ := 1

/-- The cost per pound of green tea in July -/
def july_green_tea_price : ℝ := 0.1

/-- The cost per pound of coffee in July -/
def july_coffee_price : ℝ := 2 * june_price

/-- The cost of 3 lbs of mixture containing equal quantities of green tea and coffee in July -/
def mixture_cost : ℝ := 3.15

theorem green_tea_price_decrease_proof :
  green_tea_price_decrease = (june_price - july_green_tea_price) / june_price * 100 ∧
  mixture_cost = 1.5 * july_green_tea_price + 1.5 * july_coffee_price :=
sorry

end green_tea_price_decrease_proof_l116_11630


namespace sqrt_expressions_simplification_l116_11687

theorem sqrt_expressions_simplification :
  (∀ (x y : ℝ), x > 0 → y > 0 → (Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y)) →
  (Real.sqrt 45 + Real.sqrt 50) - (Real.sqrt 18 - Real.sqrt 20) = 5 * Real.sqrt 5 + 2 * Real.sqrt 2 ∧
  Real.sqrt 24 / (6 * Real.sqrt (1/6)) - Real.sqrt 12 * (Real.sqrt 3 / 2) = -1 := by
  sorry

end sqrt_expressions_simplification_l116_11687


namespace min_distance_ant_spider_l116_11645

/-- The minimum distance between a point on the unit circle and a corresponding point on the x-axis -/
theorem min_distance_ant_spider :
  let f : ℝ → ℝ := λ a => Real.sqrt ((a - (1 - 2*a))^2 + (Real.sqrt (1 - a^2))^2)
  ∃ a : ℝ, ∀ x : ℝ, f x ≥ f a ∧ f a = Real.sqrt 14 / 4 := by
  sorry

end min_distance_ant_spider_l116_11645


namespace all_cloaks_still_too_short_l116_11677

/-- Represents a knight with a height and a cloak length -/
structure Knight where
  height : ℝ
  cloakLength : ℝ

/-- Predicate to check if a cloak is too short for a knight -/
def isCloakTooShort (k : Knight) : Prop := k.cloakLength < k.height

/-- Function to redistribute cloaks -/
def redistributeCloaks (knights : List Knight) : List Knight :=
  sorry

theorem all_cloaks_still_too_short (knights : List Knight) 
  (h1 : knights.length = 20)
  (h2 : ∀ k ∈ knights, isCloakTooShort k)
  (h3 : List.Pairwise (λ k1 k2 => k1.height ≤ k2.height) knights)
  : ∀ k ∈ redistributeCloaks knights, isCloakTooShort k :=
by sorry

end all_cloaks_still_too_short_l116_11677


namespace function_bound_l116_11654

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 13

-- State the theorem
theorem function_bound (x m : ℝ) (h : |x - m| < 1) : |f x - f m| < 2 * (|m| + 1) := by
  sorry

end function_bound_l116_11654


namespace g_of_5_l116_11650

def g (x : ℝ) : ℝ := 3*x^4 - 8*x^3 + 15*x^2 - 10*x - 75

theorem g_of_5 : g 5 = 1125 := by
  sorry

end g_of_5_l116_11650


namespace missing_legos_l116_11635

theorem missing_legos (total : ℕ) (in_box : ℕ) : 
  total = 500 → in_box = 245 → (total / 2 - in_box : ℤ) = 5 := by
  sorry

end missing_legos_l116_11635


namespace cars_cannot_meet_between_intersections_l116_11604

/-- Represents a point in the triangular grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a direction in the triangular grid --/
inductive Direction
  | Up
  | UpRight
  | DownRight

/-- Represents a car's state --/
structure CarState where
  position : GridPoint
  direction : Direction

/-- Represents the possible moves a car can make --/
inductive Move
  | Straight
  | Left
  | Right

/-- Function to update a car's state based on a move --/
def updateCarState (state : CarState) (move : Move) : CarState :=
  sorry

/-- Predicate to check if two cars are at the same position --/
def samePosition (car1 : CarState) (car2 : CarState) : Prop :=
  car1.position = car2.position

/-- Predicate to check if a point is an intersection --/
def isIntersection (point : GridPoint) : Prop :=
  sorry

/-- Theorem stating that two cars cannot meet between intersections --/
theorem cars_cannot_meet_between_intersections 
  (initialState : CarState) 
  (moves1 moves2 : List Move) : 
  let finalState1 := moves1.foldl updateCarState initialState
  let finalState2 := moves2.foldl updateCarState initialState
  samePosition finalState1 finalState2 → isIntersection finalState1.position :=
sorry

end cars_cannot_meet_between_intersections_l116_11604


namespace perp_planes_sufficient_not_necessary_l116_11668

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem perp_planes_sufficient_not_necessary 
  (α β : Plane) (m : Line) 
  (h_m_in_α : line_in_plane m α) :
  (∀ α β m, perp_planes α β → line_in_plane m α → perp_line_plane m β) ∧ 
  (∃ α β m, line_in_plane m α ∧ perp_line_plane m β ∧ ¬perp_planes α β) :=
sorry

end perp_planes_sufficient_not_necessary_l116_11668


namespace total_cost_calculation_total_cost_proof_l116_11690

/-- Given the price of tomatoes and cabbage per kilogram, calculate the total cost of purchasing 20 kg of tomatoes and 30 kg of cabbage. -/
theorem total_cost_calculation (a b : ℝ) : ℝ :=
  let tomato_price_per_kg := a
  let cabbage_price_per_kg := b
  let tomato_quantity := 20
  let cabbage_quantity := 30
  tomato_price_per_kg * tomato_quantity + cabbage_price_per_kg * cabbage_quantity

#check total_cost_calculation

theorem total_cost_proof (a b : ℝ) :
  total_cost_calculation a b = 20 * a + 30 * b := by
  sorry

end total_cost_calculation_total_cost_proof_l116_11690


namespace tangent_length_specific_tangent_length_l116_11633

/-- Given a circle with radius r, a point M at distance d from the center,
    and a line through M tangent to the circle at A, 
    the length of AM is sqrt(d^2 - r^2) -/
theorem tangent_length (r d : ℝ) (hr : r > 0) (hd : d > r) :
  let am := Real.sqrt (d^2 - r^2)
  am^2 = d^2 - r^2 := by sorry

/-- In a circle with radius 10, if a point M is 26 units away from the center
    and a line passing through M touches the circle at point A,
    then the length of AM is 24 units -/
theorem specific_tangent_length :
  let r : ℝ := 10
  let d : ℝ := 26
  let am := Real.sqrt (d^2 - r^2)
  am = 24 := by sorry

end tangent_length_specific_tangent_length_l116_11633


namespace exists_m_divisible_by_1988_l116_11620

def f (x : ℤ) : ℤ := 3 * x + 2

def iterate_f (n : ℕ) (x : ℤ) : ℤ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ+, ∃ k : ℤ, iterate_f 100 m.val = 1988 * k := by
  sorry

end exists_m_divisible_by_1988_l116_11620


namespace fifteen_percent_of_number_l116_11663

theorem fifteen_percent_of_number (x : ℝ) : 12 = 0.15 * x → x = 80 := by
  sorry

end fifteen_percent_of_number_l116_11663


namespace spinner_final_direction_l116_11694

-- Define the four cardinal directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define a function to rotate a direction
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  match (revolutions % 1).num.mod 4 with
  | 0 => d
  | 1 => match d with
         | Direction.North => Direction.East
         | Direction.East => Direction.South
         | Direction.South => Direction.West
         | Direction.West => Direction.North
  | 2 => match d with
         | Direction.North => Direction.South
         | Direction.East => Direction.West
         | Direction.South => Direction.North
         | Direction.West => Direction.East
  | 3 => match d with
         | Direction.North => Direction.West
         | Direction.East => Direction.North
         | Direction.South => Direction.East
         | Direction.West => Direction.South
  | _ => d  -- This case should never occur due to mod 4

-- Theorem statement
theorem spinner_final_direction :
  let initial_direction := Direction.North
  let clockwise_move := (7 : ℚ) / 2
  let counterclockwise_move := (17 : ℚ) / 4
  let final_direction := rotate initial_direction (clockwise_move - counterclockwise_move)
  final_direction = Direction.East := by
  sorry


end spinner_final_direction_l116_11694


namespace quadratic_transformation_l116_11640

theorem quadratic_transformation (x : ℝ) :
  x^2 - 10*x - 1 = 0 ↔ (x - 5)^2 = 26 := by sorry

end quadratic_transformation_l116_11640


namespace book_pages_count_book_pages_count_proof_l116_11678

theorem book_pages_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (days : ℕ) (avg_first_three : ℕ) (avg_next_three : ℕ) (last_day : ℕ) =>
    days = 7 →
    avg_first_three = 42 →
    avg_next_three = 39 →
    last_day = 28 →
    3 * avg_first_three + 3 * avg_next_three + last_day = 271

-- The proof is omitted
theorem book_pages_count_proof : book_pages_count 7 42 39 28 := by sorry

end book_pages_count_book_pages_count_proof_l116_11678


namespace subtraction_grouping_l116_11667

theorem subtraction_grouping (a b c d : ℝ) : a - b + c - d = a + c - (b + d) := by
  sorry

end subtraction_grouping_l116_11667


namespace area_sin_3x_l116_11627

open Real MeasureTheory

/-- The area of a function f on [a, b] -/
noncomputable def area (f : ℝ → ℝ) (a b : ℝ) : ℝ := ∫ x in a..b, f x

/-- For any positive integer n, the area of sin(nx) on [0, π/n] is 2/n -/
axiom area_sin_nx (n : ℕ+) : area (fun x ↦ sin (n * x)) 0 (π / n) = 2 / n

/-- The area of sin(3x) on [0, π/3] is 2/3 -/
theorem area_sin_3x : area (fun x ↦ sin (3 * x)) 0 (π / 3) = 2 / 3 := by
  sorry

end area_sin_3x_l116_11627


namespace binary_to_decimal_example_l116_11606

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number -/
def binary_number : List Nat := [1, 1, 0, 1, 1, 1, 1, 0, 1]

/-- Theorem stating that the given binary number is equal to 379 in decimal -/
theorem binary_to_decimal_example : binary_to_decimal binary_number = 379 := by
  sorry

end binary_to_decimal_example_l116_11606


namespace units_digit_of_product_of_first_four_composites_l116_11638

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_of_first_four_composites :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end units_digit_of_product_of_first_four_composites_l116_11638
