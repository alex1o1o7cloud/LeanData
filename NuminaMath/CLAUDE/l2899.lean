import Mathlib

namespace roller_coaster_cost_proof_l2899_289913

/-- The cost of a ride on the Ferris wheel in tickets -/
def ferris_wheel_cost : ℚ := 2

/-- The discount in tickets for going on multiple rides -/
def multiple_ride_discount : ℚ := 1

/-- The value of the newspaper coupon in tickets -/
def newspaper_coupon : ℚ := 1

/-- The total number of tickets Zach needed to buy for both rides -/
def total_tickets_bought : ℚ := 7

/-- The cost of a ride on the roller coaster in tickets -/
def roller_coaster_cost : ℚ := 7

theorem roller_coaster_cost_proof :
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - newspaper_coupon = total_tickets_bought :=
by sorry

end roller_coaster_cost_proof_l2899_289913


namespace building_heights_sum_l2899_289997

/-- The combined height of three buildings with their antennas -/
def combined_height (esb_height esb_antenna wt_height wt_antenna owt_height owt_antenna : ℕ) : ℕ :=
  (esb_height + esb_antenna) + (wt_height + wt_antenna) + (owt_height + owt_antenna)

/-- Theorem stating the combined height of the three buildings -/
theorem building_heights_sum :
  combined_height 1250 204 1450 280 1368 408 = 4960 := by
  sorry

end building_heights_sum_l2899_289997


namespace perpendicular_line_through_point_l2899_289945

/-- Given a line L1 with equation x - 2y + 3 = 0 and a point P(1, -3),
    prove that the line L2 with equation 2x + y + 1 = 0 passes through P
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => x - 2 * y + 3 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => 2 * x + y + 1 = 0
  let P : ℝ × ℝ := (1, -3)
  (L2 P.1 P.2) ∧                           -- L2 passes through P
  (∀ (x1 y1 x2 y2 : ℝ),
    L1 x1 y1 → L1 x2 y2 → x1 ≠ x2 →        -- L1 and L2 are perpendicular
    L2 x1 y1 → L2 x2 y2 → 
    (x2 - x1) * ((x2 - x1) / (y2 - y1)) = -1) :=
by sorry

end perpendicular_line_through_point_l2899_289945


namespace bicycle_helmet_cost_increase_l2899_289955

/-- The percent increase in the combined cost of a bicycle and helmet --/
theorem bicycle_helmet_cost_increase 
  (bicycle_cost : ℝ) 
  (helmet_cost : ℝ) 
  (bicycle_increase_percent : ℝ) 
  (helmet_increase_percent : ℝ) 
  (h1 : bicycle_cost = 150)
  (h2 : helmet_cost = 50)
  (h3 : bicycle_increase_percent = 10)
  (h4 : helmet_increase_percent = 20) : 
  ((bicycle_cost * (1 + bicycle_increase_percent / 100) + 
    helmet_cost * (1 + helmet_increase_percent / 100)) - 
   (bicycle_cost + helmet_cost)) / (bicycle_cost + helmet_cost) * 100 = 12.5 := by
  sorry

end bicycle_helmet_cost_increase_l2899_289955


namespace factorization_proof_l2899_289952

theorem factorization_proof (x : ℝ) : 4*x^3 - 8*x^2 + 4*x = 4*x*(x-1)^2 := by
  sorry

end factorization_proof_l2899_289952


namespace ophelia_age_l2899_289931

/-- Given the following conditions:
  1. In 10 years, Ophelia will be thrice as old as Lennon.
  2. In 10 years, Mike will be twice the age difference between Ophelia and Lennon.
  3. Lennon is currently 8 years old.
  4. Mike is currently 5 years older than Lennon.
Prove that Ophelia's current age is 44 years. -/
theorem ophelia_age (lennon_age : ℕ) (mike_age : ℕ) (ophelia_age : ℕ) :
  lennon_age = 8 →
  mike_age = lennon_age + 5 →
  ophelia_age + 10 = 3 * (lennon_age + 10) →
  mike_age + 10 = 2 * ((ophelia_age + 10) - (lennon_age + 10)) →
  ophelia_age = 44 := by
  sorry

end ophelia_age_l2899_289931


namespace price_of_first_oil_l2899_289963

/-- Given two oils mixed together, prove the price of the first oil. -/
theorem price_of_first_oil :
  -- Define the volumes of oils
  let volume_first : ℝ := 10
  let volume_second : ℝ := 5
  -- Define the price of the second oil
  let price_second : ℝ := 68
  -- Define the price of the mixture
  let price_mixture : ℝ := 56
  -- Define the total volume
  let volume_total : ℝ := volume_first + volume_second
  -- The equation that represents the mixing of oils
  ∀ price_first : ℝ,
    volume_first * price_first + volume_second * price_second =
    volume_total * price_mixture →
    -- Prove that the price of the first oil is 50
    price_first = 50 := by
  sorry

end price_of_first_oil_l2899_289963


namespace x_convergence_bound_l2899_289993

def x : ℕ → ℚ
  | 0 => 5
  | n + 1 => (x n ^ 2 + 5 * x n + 4) / (x n + 6)

theorem x_convergence_bound :
  ∃ m : ℕ, (∀ k < m, x k > 4 + 1 / 2^20) ∧
           (x m ≤ 4 + 1 / 2^20) ∧
           (81 ≤ m) ∧ (m ≤ 242) :=
by sorry

end x_convergence_bound_l2899_289993


namespace inequality_solution_quadratic_inequality_l2899_289998

-- Part 1
def solution_set (x : ℝ) : Prop :=
  x ∈ (Set.Iic (-4) ∪ Set.Ici (1/2))

theorem inequality_solution :
  ∀ x : ℝ, (9 / (x + 4) ≤ 2) ↔ solution_set x := by sorry

-- Part 2
def valid_k (k : ℝ) : Prop :=
  k ∈ (Set.Iio (-Real.sqrt 2) ∪ Set.Ioi (Real.sqrt 2))

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 1 > 0) → valid_k k := by sorry

end inequality_solution_quadratic_inequality_l2899_289998


namespace tangent_line_intersection_l2899_289970

/-- A line l passes through point A(t,0) and is tangent to the curve y = x^2 with an angle of inclination of 45° -/
theorem tangent_line_intersection (t : ℝ) : 
  (∃ (m : ℝ), 
    -- The line passes through (t, 0)
    (t - m) * (m^2 - 0) = (1 - 0) * (0 - t) ∧ 
    -- The line is tangent to y = x^2 at (m, m^2)
    2 * m = 1 ∧ 
    -- The angle of inclination is 45°
    (m^2 - 0) / (m - t) = 1) → 
  t = 1/4 := by sorry

end tangent_line_intersection_l2899_289970


namespace special_line_equation_l2899_289903

/-- A line passing through (-10, 10) with x-intercept four times y-intercept -/
structure SpecialLine where
  -- The line passes through (-10, 10)
  passes_through : (Int × Int)
  -- The x-intercept is four times the y-intercept
  x_intercept_relation : ℝ → ℝ → Prop

/-- The equation of the special line -/
def line_equation (L : SpecialLine) : Set (ℝ × ℝ) :=
  {(x, y) | x + y = 0 ∨ x + 4*y - 30 = 0}

/-- Theorem stating that the equation of the special line is correct -/
theorem special_line_equation (L : SpecialLine) 
  (h1 : L.passes_through = (-10, 10))
  (h2 : L.x_intercept_relation = λ x y => x = 4*y) :
  ∀ (x y : ℝ), (x, y) ∈ line_equation L ↔ 
    ((x + y = 0) ∨ (x + 4*y - 30 = 0)) :=
by
  sorry


end special_line_equation_l2899_289903


namespace cakes_served_during_lunch_l2899_289968

/-- Given that a restaurant served some cakes for lunch and dinner, with a total of 15 cakes served today and 9 cakes served during dinner, prove that 6 cakes were served during lunch. -/
theorem cakes_served_during_lunch (total_cakes dinner_cakes lunch_cakes : ℕ) 
  (h1 : total_cakes = 15)
  (h2 : dinner_cakes = 9)
  (h3 : total_cakes = lunch_cakes + dinner_cakes) :
  lunch_cakes = 6 := by
  sorry

end cakes_served_during_lunch_l2899_289968


namespace cindy_marbles_l2899_289925

theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ) :
  initial_marbles = 500 →
  friends = 4 →
  marbles_per_friend = 80 →
  4 * (initial_marbles - friends * marbles_per_friend) = 720 :=
by sorry

end cindy_marbles_l2899_289925


namespace smallest_divisor_for_perfect_square_l2899_289999

theorem smallest_divisor_for_perfect_square (n : ℕ) (h : n = 2880) :
  ∃ (d : ℕ), d > 0 ∧ d.min = 5 ∧ (∃ (k : ℕ), n / d = k * k) ∧
  ∀ (x : ℕ), 0 < x ∧ x < d → ¬∃ (m : ℕ), n / x = m * m :=
sorry

end smallest_divisor_for_perfect_square_l2899_289999


namespace naples_pizza_weight_l2899_289996

/-- The total weight of pizza eaten by Rachel and Bella in Naples -/
def total_pizza_weight (rachel_pizza : ℕ) (rachel_mushrooms : ℕ) (rachel_olives : ℕ)
                       (bella_pizza : ℕ) (bella_cheese : ℕ) (bella_onions : ℕ) : ℕ :=
  (rachel_pizza + rachel_mushrooms + rachel_olives) + (bella_pizza + bella_cheese + bella_onions)

/-- Theorem stating the total weight of pizza eaten by Rachel and Bella in Naples -/
theorem naples_pizza_weight :
  total_pizza_weight 598 100 50 354 75 55 = 1232 := by
  sorry

end naples_pizza_weight_l2899_289996


namespace function_inequality_l2899_289959

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem function_inequality (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : ∀ x, f (x + 2) = -f x)
  (h_decreasing : is_decreasing_on f 0 1) :
  f (3/2) < f (1/4) ∧ f (1/4) < f (-1/4) := by
  sorry

end function_inequality_l2899_289959


namespace inequality_proof_l2899_289991

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l2899_289991


namespace gumball_difference_l2899_289908

theorem gumball_difference (carolyn lew amanda tom : ℕ) (x : ℕ) :
  carolyn = 17 →
  lew = 12 →
  amanda = 24 →
  tom = 8 →
  14 ≤ (carolyn + lew + amanda + tom + x) / 7 →
  (carolyn + lew + amanda + tom + x) / 7 ≤ 32 →
  ∃ (x_min x_max : ℕ), x_min ≤ x ∧ x ≤ x_max ∧ x_max - x_min = 126 :=
by sorry

end gumball_difference_l2899_289908


namespace product_equals_zero_l2899_289971

theorem product_equals_zero (b : ℤ) (h : b = 3) : 
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end product_equals_zero_l2899_289971


namespace sum_of_specific_numbers_l2899_289973

theorem sum_of_specific_numbers (a b : ℝ) 
  (ha_abs : |a| = 5)
  (hb_abs : |b| = 2)
  (ha_neg : a < 0)
  (hb_pos : b > 0) :
  a + b = -3 := by
sorry

end sum_of_specific_numbers_l2899_289973


namespace min_value_quadratic_l2899_289982

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 217 ∧ ∀ x, 3 * x^2 - 18 * x + 244 ≥ m := by
  sorry

end min_value_quadratic_l2899_289982


namespace cao_required_proof_l2899_289917

/-- Represents the balanced chemical equation for the reaction between Calcium oxide and Water to form Calcium hydroxide -/
structure BalancedEquation where
  cao : ℕ
  h2o : ℕ
  caoh2 : ℕ
  balanced : cao = h2o ∧ cao = caoh2

/-- Calculates the required amount of Calcium oxide given the amounts of Water and Calcium hydroxide -/
def calcCaORequired (water : ℕ) (hydroxide : ℕ) (eq : BalancedEquation) : ℕ :=
  if water = hydroxide then water else 0

theorem cao_required_proof (water : ℕ) (hydroxide : ℕ) (eq : BalancedEquation) 
  (h1 : water = 3) 
  (h2 : hydroxide = 3) : 
  calcCaORequired water hydroxide eq = 3 := by
  sorry

end cao_required_proof_l2899_289917


namespace exchange_impossibility_l2899_289961

theorem exchange_impossibility : 
  ¬∃ (x y z : ℕ), x + y + z = 10 ∧ x + 3*y + 5*z = 25 :=
sorry

end exchange_impossibility_l2899_289961


namespace sum_and_reciprocal_geq_two_l2899_289975

theorem sum_and_reciprocal_geq_two (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end sum_and_reciprocal_geq_two_l2899_289975


namespace count_80_in_scores_l2899_289984

def scores : List ℕ := [80, 90, 80, 80, 100, 70]

theorem count_80_in_scores : (scores.filter (· = 80)).length = 3 := by
  sorry

end count_80_in_scores_l2899_289984


namespace cube_cross_section_area_l2899_289960

/-- The area of a cross-section in a cube -/
theorem cube_cross_section_area (a : ℝ) (h : a > 0) :
  let cube_edge := a
  let face_diagonal := a * Real.sqrt 2
  let space_diagonal := a * Real.sqrt 3
  let cross_section_area := (face_diagonal * space_diagonal) / 2
  cross_section_area = (a^2 * Real.sqrt 6) / 2 := by
sorry

end cube_cross_section_area_l2899_289960


namespace inequality_range_l2899_289906

theorem inequality_range (a : ℝ) : 
  (∀ (x θ : ℝ), 0 ≤ θ ∧ θ ≤ Real.pi / 2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
by sorry

end inequality_range_l2899_289906


namespace quadratic_equation_roots_l2899_289972

theorem quadratic_equation_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁^2 - 2*r₁ = 0 ∧ r₂^2 - 2*r₂ = 0) :=
by sorry

end quadratic_equation_roots_l2899_289972


namespace angle_D_measure_l2899_289901

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- Measure of angle E in degrees -/
  angleE : ℝ
  /-- The triangle is isosceles with angle D congruent to angle F -/
  isIsosceles : True
  /-- The measure of angle F is three times the measure of angle E -/
  angleFRelation : True

/-- The measure of angle D in the isosceles triangle -/
def measureAngleD (t : IsoscelesTriangle) : ℝ :=
  3 * t.angleE

/-- Theorem: The measure of angle D is approximately 77 degrees -/
theorem angle_D_measure (t : IsoscelesTriangle) :
  ‖measureAngleD t - 77‖ < 1 := by
  sorry

#check angle_D_measure

end angle_D_measure_l2899_289901


namespace smallest_of_five_consecutive_even_l2899_289940

def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

def sum_five_consecutive_even (k : ℕ) : ℕ := 5 * (2 * k + 2)

theorem smallest_of_five_consecutive_even : 
  ∃ k : ℕ, sum_five_consecutive_even k = sum_first_n_even 25 ∧ 
  2 * k + 2 = 126 := by
  sorry

end smallest_of_five_consecutive_even_l2899_289940


namespace largest_non_expressible_l2899_289989

def is_non_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def is_expressible (n : ℕ) : Prop :=
  ∃ (k m : ℕ), k > 0 ∧ is_non_prime m ∧ n = 47 * k + m

theorem largest_non_expressible : 
  (∀ n > 90, is_expressible n) ∧ 
  ¬is_expressible 90 ∧
  (∀ n < 90, ¬is_expressible n → n < 90) :=
sorry

end largest_non_expressible_l2899_289989


namespace electronics_store_cost_l2899_289946

/-- Given the cost of 5 MP3 players and 8 headphones is $840, and the cost of one set of headphones
is $30, prove that the cost of 3 MP3 players and 4 headphones is $480. -/
theorem electronics_store_cost (mp3_cost headphones_cost : ℕ) : 
  5 * mp3_cost + 8 * headphones_cost = 840 →
  headphones_cost = 30 →
  3 * mp3_cost + 4 * headphones_cost = 480 := by
  sorry

end electronics_store_cost_l2899_289946


namespace intersection_nonempty_iff_m_in_range_l2899_289980

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m / 2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

def B (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2 * m + 1}

-- State the theorem
theorem intersection_nonempty_iff_m_in_range (m : ℝ) :
  (A m ∩ B m).Nonempty ↔ 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 := by
  sorry


end intersection_nonempty_iff_m_in_range_l2899_289980


namespace drama_club_pets_l2899_289938

theorem drama_club_pets (S : Finset ℕ) (R G : Finset ℕ) : 
  Finset.card S = 50 → 
  (∀ s ∈ S, s ∈ R ∨ s ∈ G) → 
  Finset.card R = 35 → 
  Finset.card G = 40 → 
  Finset.card (R ∩ G) = 25 := by
sorry

end drama_club_pets_l2899_289938


namespace polynomial_piecewise_function_l2899_289985

theorem polynomial_piecewise_function 
  (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = 0)
  (hg : ∀ x, g x = -x)
  (hh : ∀ x, h x = -x + 2) :
  ∀ x, |f x| - |g x| + h x = 
    if x < -1 then -1
    else if x ≤ 0 then 2
    else -2*x + 2 := by
  sorry

end polynomial_piecewise_function_l2899_289985


namespace crabapple_sequences_l2899_289927

/-- The number of students in the class -/
def num_students : ℕ := 11

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 4

/-- The number of possible sequences of crabapple recipients in a week -/
def possible_sequences : ℕ := num_students ^ meetings_per_week

theorem crabapple_sequences :
  possible_sequences = 14641 :=
sorry

end crabapple_sequences_l2899_289927


namespace food_additives_budget_percentage_l2899_289964

theorem food_additives_budget_percentage :
  ∀ (total_degrees : ℝ) 
    (microphotonics_percent : ℝ) 
    (home_electronics_percent : ℝ) 
    (genetically_modified_microorganisms_percent : ℝ) 
    (industrial_lubricants_percent : ℝ) 
    (basic_astrophysics_degrees : ℝ),
  total_degrees = 360 →
  microphotonics_percent = 14 →
  home_electronics_percent = 19 →
  genetically_modified_microorganisms_percent = 24 →
  industrial_lubricants_percent = 8 →
  basic_astrophysics_degrees = 90 →
  ∃ (food_additives_percent : ℝ),
    food_additives_percent = 10 ∧
    microphotonics_percent + home_electronics_percent + 
    genetically_modified_microorganisms_percent + industrial_lubricants_percent +
    (basic_astrophysics_degrees / total_degrees * 100) + food_additives_percent = 100 :=
by sorry

end food_additives_budget_percentage_l2899_289964


namespace blocks_used_for_tower_l2899_289950

theorem blocks_used_for_tower (initial_blocks : ℕ) (blocks_left : ℕ) : 
  initial_blocks = 78 → blocks_left = 59 → initial_blocks - blocks_left = 19 := by
  sorry

end blocks_used_for_tower_l2899_289950


namespace men_in_first_group_l2899_289969

/-- The number of days taken by the first group to complete the job -/
def days_group1 : ℕ := 15

/-- The number of men in the second group -/
def men_group2 : ℕ := 25

/-- The number of days taken by the second group to complete the job -/
def days_group2 : ℕ := 24

/-- The amount of work done is the product of the number of workers and the number of days they work -/
def work_done (men : ℕ) (days : ℕ) : ℕ := men * days

/-- The theorem stating that the number of men in the first group is 40 -/
theorem men_in_first_group : 
  ∃ (men_group1 : ℕ), 
    men_group1 = 40 ∧ 
    work_done men_group1 days_group1 = work_done men_group2 days_group2 :=
by sorry

end men_in_first_group_l2899_289969


namespace special_quadrilateral_AD_length_special_quadrilateral_AD_length_is_30_l2899_289948

/-- A quadrilateral with specific side lengths and angle properties -/
structure SpecialQuadrilateral where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CD : ℝ
  -- Angle properties
  B_obtuse : ℝ
  C_obtuse : ℝ
  sin_C : ℝ
  cos_B : ℝ
  -- Conditions
  AB_eq : AB = 6
  BC_eq : BC = 8
  CD_eq : CD = 15
  B_obtuse_cond : B_obtuse > π / 2
  C_obtuse_cond : C_obtuse > π / 2
  sin_C_eq : sin_C = 4 / 5
  cos_B_eq : cos_B = -4 / 5

/-- The length of side AD in the special quadrilateral is 30 -/
theorem special_quadrilateral_AD_length (q : SpecialQuadrilateral) : ℝ :=
  30

/-- The main theorem stating that for any special quadrilateral, 
    the length of side AD is 30 -/
theorem special_quadrilateral_AD_length_is_30 (q : SpecialQuadrilateral) :
  special_quadrilateral_AD_length q = 30 := by
  sorry

end special_quadrilateral_AD_length_special_quadrilateral_AD_length_is_30_l2899_289948


namespace kim_payroll_time_l2899_289918

/-- Represents the time Kim spends on her morning routine -/
structure MorningRoutine where
  coffee_time : ℕ
  status_update_time_per_employee : ℕ
  num_employees : ℕ
  total_routine_time : ℕ

/-- Calculates the time spent per employee on payroll records -/
def payroll_time_per_employee (routine : MorningRoutine) : ℕ :=
  let total_status_update_time := routine.status_update_time_per_employee * routine.num_employees
  let remaining_time := routine.total_routine_time - (routine.coffee_time + total_status_update_time)
  remaining_time / routine.num_employees

/-- Theorem stating that Kim spends 3 minutes per employee updating payroll records -/
theorem kim_payroll_time (kim_routine : MorningRoutine) 
  (h1 : kim_routine.coffee_time = 5)
  (h2 : kim_routine.status_update_time_per_employee = 2)
  (h3 : kim_routine.num_employees = 9)
  (h4 : kim_routine.total_routine_time = 50) :
  payroll_time_per_employee kim_routine = 3 := by
  sorry

#eval payroll_time_per_employee { coffee_time := 5, status_update_time_per_employee := 2, num_employees := 9, total_routine_time := 50 }

end kim_payroll_time_l2899_289918


namespace ratio_of_percentages_l2899_289953

theorem ratio_of_percentages (P Q M N R : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hN : N = 0.6 * P)
  (hR : R = 0.3 * N)
  (hP : P ≠ 0) : 
  M / R = 5 / 9 := by
  sorry

end ratio_of_percentages_l2899_289953


namespace no_adjacent_standing_probability_l2899_289904

-- Define the number of people
def n : ℕ := 10

-- Define a function to represent the recursive relation
def a : ℕ → ℕ
| 0 => 1
| 1 => 2
| m + 2 => a (m + 1) + a m

-- Define the probability
def probability : ℚ := a n / 2^n

-- Theorem statement
theorem no_adjacent_standing_probability : probability = 123 / 1024 := by
  sorry

end no_adjacent_standing_probability_l2899_289904


namespace parabola_focus_directrix_distance_l2899_289987

/-- Given a parabola and a circle with specific properties, 
    prove the distance from the focus of the parabola to its directrix -/
theorem parabola_focus_directrix_distance 
  (p : ℝ) 
  (h_p_pos : p > 0)
  (h_intersect : ∃ (A B : ℝ × ℝ), 
    A ≠ B ∧
    A.2^2 = 2*p*A.1 ∧ 
    B.2^2 = 2*p*B.1 ∧
    A.1^2 + (A.2 - 1)^2 = 1 ∧
    B.1^2 + (B.2 - 1)^2 = 1)
  (h_distance : ∃ (A B : ℝ × ℝ), 
    A ≠ B ∧
    A.2^2 = 2*p*A.1 ∧ 
    B.2^2 = 2*p*B.1 ∧
    A.1^2 + (A.2 - 1)^2 = 1 ∧
    B.1^2 + (B.2 - 1)^2 = 1 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4/3) :
  p = Real.sqrt 2 / 6 := by
    sorry

end parabola_focus_directrix_distance_l2899_289987


namespace nancy_water_intake_percentage_l2899_289928

/-- Given Nancy's daily water intake and body weight, calculate the percentage of her body weight she drinks in water. -/
theorem nancy_water_intake_percentage 
  (daily_water_intake : ℝ) 
  (body_weight : ℝ) 
  (h1 : daily_water_intake = 54) 
  (h2 : body_weight = 90) : 
  (daily_water_intake / body_weight) * 100 = 60 := by
  sorry

end nancy_water_intake_percentage_l2899_289928


namespace variance_of_transformed_binomial_l2899_289905

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Variance of a linear transformation of a random variable -/
def variance_linear_transform (X : BinomialRV) (a b : ℝ) : ℝ := a^2 * variance X

/-- Main theorem: Variance of 4ξ + 3 for ξ ~ B(100, 0.2) -/
theorem variance_of_transformed_binomial :
  ∃ (ξ : BinomialRV), ξ.n = 100 ∧ ξ.p = 0.2 ∧ variance_linear_transform ξ 4 3 = 256 := by
  sorry

end variance_of_transformed_binomial_l2899_289905


namespace xy_value_l2899_289994

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := by
  sorry

end xy_value_l2899_289994


namespace least_consecutive_bigness_l2899_289924

def bigness (a b c : ℕ) : ℕ := a * b * c + 2 * (a * b + b * c + a * c) + 4 * (a + b + c)

def has_integer_sides (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ bigness a b c = n

theorem least_consecutive_bigness :
  (∀ k < 55, ¬(has_integer_sides k ∧ has_integer_sides (k + 1))) ∧
  (has_integer_sides 55 ∧ has_integer_sides 56) :=
sorry

end least_consecutive_bigness_l2899_289924


namespace triangles_on_ABC_l2899_289942

/-- The number of triangles that can be formed with marked points on the sides of a triangle -/
def num_triangles (points_AB points_BC points_AC : ℕ) : ℕ :=
  let total_points := points_AB + points_BC + points_AC
  let total_combinations := (total_points.choose 3)
  let invalid_combinations := (points_AB.choose 3) + (points_BC.choose 3) + (points_AC.choose 3)
  total_combinations - invalid_combinations

/-- Theorem stating the number of triangles formed with marked points on triangle ABC -/
theorem triangles_on_ABC : num_triangles 12 9 10 = 4071 := by
  sorry

end triangles_on_ABC_l2899_289942


namespace arithmetic_sequence_sum_l2899_289907

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 39 →
  a 2 + a 5 + a 8 = 33 →
  a 3 + a 6 + a 9 = 27 := by
sorry

end arithmetic_sequence_sum_l2899_289907


namespace union_of_A_and_B_l2899_289949

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x > -1} := by sorry

end union_of_A_and_B_l2899_289949


namespace inequality_system_solution_l2899_289921

theorem inequality_system_solution :
  let S : Set ℝ := {x | 5 * x - 2 < 3 * (x + 1) ∧ (3 * x - 2) / 3 ≥ x + (x - 2) / 2}
  S = {x | x ≤ 2/3} := by
sorry

end inequality_system_solution_l2899_289921


namespace rectangle_cut_theorem_l2899_289935

/-- In a rectangle with a line cutting it, if the area of the resulting quadrilateral
    is 40% of the total area, then the height of this quadrilateral is 0.8 times
    the length of the rectangle. -/
theorem rectangle_cut_theorem (L W x : ℝ) : 
  L > 0 → W > 0 → x > 0 →
  x * W / 2 = 0.4 * L * W →
  x = 0.8 * L := by
sorry

end rectangle_cut_theorem_l2899_289935


namespace ballCombinations_2005_l2899_289974

/-- The number of ways to choose n balls from red, green, and yellow balls
    such that the number of red balls is even or the number of green balls is odd. -/
def ballCombinations (n : ℕ) : ℕ := sorry

/-- The main theorem stating the number of combinations for 2005 balls. -/
theorem ballCombinations_2005 : ballCombinations 2005 = Nat.choose 2007 2 - Nat.choose 1004 2 := by
  sorry

end ballCombinations_2005_l2899_289974


namespace quadratic_inequality_range_l2899_289911

theorem quadratic_inequality_range (x : ℝ) (h : x^2 - 3*x + 2 < 0) :
  ∃ y ∈ Set.Ioo (-0.25 : ℝ) 0, y = x^2 - 3*x + 2 :=
sorry

end quadratic_inequality_range_l2899_289911


namespace sum_with_radical_conjugate_l2899_289986

theorem sum_with_radical_conjugate :
  let a := 15 - Real.sqrt 500
  let radical_conjugate := 15 + Real.sqrt 500
  a + radical_conjugate = 30 := by sorry

end sum_with_radical_conjugate_l2899_289986


namespace quadratic_equal_roots_l2899_289920

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + m*y + m = 0 → y = x) ↔ 
  (m = 0 ∨ m = 4) :=
sorry

end quadratic_equal_roots_l2899_289920


namespace piggy_bank_dime_difference_l2899_289923

theorem piggy_bank_dime_difference :
  ∀ (a b c d : ℕ),
  a + b + c + d = 150 →
  5 * a + 10 * b + 25 * c + 50 * d = 1500 →
  (∃ (b_max b_min : ℕ),
    (∀ (a' b' c' d' : ℕ),
      a' + b' + c' + d' = 150 →
      5 * a' + 10 * b' + 25 * c' + 50 * d' = 1500 →
      b' ≤ b_max ∧ b' ≥ b_min) ∧
    b_max - b_min = 150) :=
by sorry

end piggy_bank_dime_difference_l2899_289923


namespace stock_bond_relationship_l2899_289992

/-- Represents the investment portfolio of Matthew -/
structure Portfolio where
  expensive_stock_value : ℝ  -- Value of the more expensive stock per share
  expensive_stock_shares : ℕ  -- Number of shares of the more expensive stock
  cheap_stock_shares : ℕ     -- Number of shares of the cheaper stock
  bond_face_value : ℝ        -- Face value of the bond
  bond_coupon_rate : ℝ       -- Annual coupon rate of the bond
  bond_discount : ℝ          -- Discount rate at which the bond was purchased
  total_assets : ℝ           -- Total value of assets in stocks and bond
  bond_market_value : ℝ      -- Current market value of the bond

/-- Theorem stating the relationship between the more expensive stock value and the bond market value -/
theorem stock_bond_relationship (p : Portfolio) 
  (h1 : p.expensive_stock_shares = 14)
  (h2 : p.cheap_stock_shares = 26)
  (h3 : p.bond_face_value = 1000)
  (h4 : p.bond_coupon_rate = 0.06)
  (h5 : p.bond_discount = 0.03)
  (h6 : p.total_assets = 2106)
  (h7 : p.expensive_stock_value * p.expensive_stock_shares + 
        (p.expensive_stock_value / 2) * p.cheap_stock_shares + 
        p.bond_market_value = p.total_assets) :
  p.bond_market_value = 2106 - 27 * p.expensive_stock_value := by
  sorry

end stock_bond_relationship_l2899_289992


namespace sin_graph_transform_l2899_289965

/-- Given a function f : ℝ → ℝ where f x = sin x for all x ∈ ℝ,
    prove that shifting its graph left by π/3 and then halving the x-coordinates
    results in the function g where g x = sin(2x + π/3) for all x ∈ ℝ. -/
theorem sin_graph_transform (f g : ℝ → ℝ) (h₁ : ∀ x, f x = Real.sin x) 
  (h₂ : ∀ x, g x = f (2*x + π/3)) : ∀ x, g x = Real.sin (2*x + π/3) := by
  sorry

end sin_graph_transform_l2899_289965


namespace equation_is_parabola_l2899_289995

-- Define the equation
def equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 4)^2 + (y - 1)^2)

-- Theorem statement
theorem equation_is_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ x y : ℝ, equation x y ↔ y = a * x^2 + b * x + c) :=
sorry

end equation_is_parabola_l2899_289995


namespace table_tennis_equipment_theorem_l2899_289944

/-- Represents the price of table tennis equipment and store discounts -/
structure TableTennisEquipment where
  racket_price : ℝ
  ball_price : ℝ
  store_a_discount : ℝ
  store_b_free_balls : ℕ

/-- Proves that given the conditions, the prices are correct and Store A is more cost-effective -/
theorem table_tennis_equipment_theorem (e : TableTennisEquipment)
  (h1 : 2 * e.racket_price + 3 * e.ball_price = 75)
  (h2 : 3 * e.racket_price + 2 * e.ball_price = 100)
  (h3 : e.store_a_discount = 0.1)
  (h4 : e.store_b_free_balls = 10) :
  e.racket_price = 30 ∧
  e.ball_price = 5 ∧
  (1 - e.store_a_discount) * (20 * e.racket_price + 30 * e.ball_price) <
  20 * e.racket_price + (30 - e.store_b_free_balls) * e.ball_price := by
  sorry

end table_tennis_equipment_theorem_l2899_289944


namespace sum_equals_product_implies_two_greater_than_one_l2899_289981

theorem sum_equals_product_implies_two_greater_than_one 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum_prod : a + b + c = a * b * c) : 
  (a > 1 ∧ b > 1) ∨ (a > 1 ∧ c > 1) ∨ (b > 1 ∧ c > 1) := by
  sorry

end sum_equals_product_implies_two_greater_than_one_l2899_289981


namespace sin_cos_identity_l2899_289916

theorem sin_cos_identity : 
  Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_identity_l2899_289916


namespace solution_equivalence_l2899_289976

theorem solution_equivalence (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (x^2 + 3*x + 1/(x-1) = a + 1/(x-1) ↔ x^2 + 3*x = a)) ∧
  (a = 4 → ∃ x : ℝ, x^2 + 3*x = a ∧ x = 1) := by
  sorry

end solution_equivalence_l2899_289976


namespace existence_of_solution_l2899_289922

theorem existence_of_solution (p : Nat) (hp : Nat.Prime p) (hodd : Odd p) :
  ∃ (x y z t : Nat), x^2 + y^2 + z^2 = t * p ∧ t < p ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ t ≠ 0) := by
  sorry

end existence_of_solution_l2899_289922


namespace largest_integer_not_exceeding_700pi_l2899_289951

theorem largest_integer_not_exceeding_700pi :
  ⌊700 * Real.pi⌋ = 2199 := by sorry

end largest_integer_not_exceeding_700pi_l2899_289951


namespace circle_radius_l2899_289934

theorem circle_radius (x y : ℝ) (h : x + 2*y = 100*Real.pi) : 
  x = Real.pi * 8^2 ∧ y = 2 * Real.pi * 8 := by
  sorry

end circle_radius_l2899_289934


namespace sum_of_units_digits_of_seven_powers_l2899_289978

def units_digit (n : ℕ) : ℕ := n % 10

def A (n : ℕ) : ℕ := units_digit (7^n)

theorem sum_of_units_digits_of_seven_powers : 
  (Finset.range 2013).sum A + A 2013 = 10067 := by sorry

end sum_of_units_digits_of_seven_powers_l2899_289978


namespace distance_45N_90long_diff_l2899_289943

/-- The spherical distance between two points on Earth --/
def spherical_distance (R : ℝ) (lat1 lat2 long1 long2 : ℝ) : ℝ := sorry

/-- Theorem: The spherical distance between two points at 45°N with 90° longitude difference --/
theorem distance_45N_90long_diff (R : ℝ) :
  spherical_distance R (π/4) (π/4) (π/9) (11*π/18) = π*R/3 := by sorry

end distance_45N_90long_diff_l2899_289943


namespace like_terms_exponents_l2899_289962

/-- Given two algebraic expressions that are like terms, prove the values of their exponents. -/
theorem like_terms_exponents (a b : ℤ) : 
  (∀ (x y : ℝ), ∃ (k : ℝ), 2 * x^a * y^2 = k * (-3 * x^3 * y^(b+3))) → 
  (a = 3 ∧ b = -1) := by
  sorry

end like_terms_exponents_l2899_289962


namespace least_period_is_twelve_l2899_289966

/-- A function satisfying the given condition -/
def SatisfiesCondition (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) + g (x - 2) = g x

/-- The period of a function -/
def IsPeriod (g : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, g (x + p) = g x

/-- The least positive period of a function -/
def IsLeastPositivePeriod (g : ℝ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ IsPeriod g q ∧ ∀ p, 0 < p ∧ p < q → ¬IsPeriod g p

theorem least_period_is_twelve :
  ∀ g : ℝ → ℝ, SatisfiesCondition g → IsLeastPositivePeriod g 12 := by
  sorry

end least_period_is_twelve_l2899_289966


namespace larger_integer_is_nine_l2899_289939

theorem larger_integer_is_nine (x y : ℤ) (h_product : x * y = 36) (h_sum : x + y = 13) :
  max x y = 9 := by
  sorry

end larger_integer_is_nine_l2899_289939


namespace bruce_bags_l2899_289957

/-- The number of bags Bruce can buy with his change after purchasing crayons, books, and calculators. -/
def bags_bruce_can_buy (crayons_packs : ℕ) (crayons_price : ℕ) 
                       (books : ℕ) (books_price : ℕ)
                       (calculators : ℕ) (calculators_price : ℕ)
                       (initial_money : ℕ) (bag_price : ℕ) : ℕ :=
  let total_spent := crayons_packs * crayons_price + books * books_price + calculators * calculators_price
  let change := initial_money - total_spent
  change / bag_price

theorem bruce_bags : 
  bags_bruce_can_buy 5 5 10 5 3 5 200 10 = 11 := by
  sorry

end bruce_bags_l2899_289957


namespace max_leftover_fruits_l2899_289926

theorem max_leftover_fruits (A G : ℕ) : 
  (A % 7 ≤ 6) ∧ (G % 7 ≤ 6) ∧ 
  ∃ (A₀ G₀ : ℕ), A₀ % 7 = 6 ∧ G₀ % 7 = 6 :=
by sorry

end max_leftover_fruits_l2899_289926


namespace whitewashing_cost_is_2718_l2899_289967

/-- Calculate the cost of whitewashing a room's walls --/
def whitewashingCost (roomLength roomWidth roomHeight : ℝ) 
                     (doorHeight doorWidth : ℝ)
                     (windowHeight windowWidth : ℝ)
                     (numWindows : ℕ)
                     (costPerSquareFoot : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength * roomHeight + roomWidth * roomHeight)
  let doorArea := doorHeight * doorWidth
  let windowArea := windowHeight * windowWidth * numWindows
  (wallArea - doorArea - windowArea) * costPerSquareFoot

/-- Theorem stating the cost of whitewashing for the given room --/
theorem whitewashing_cost_is_2718 :
  whitewashingCost 25 15 12 6 3 4 3 3 3 = 2718 := by
  sorry

end whitewashing_cost_is_2718_l2899_289967


namespace cost_of_500_pieces_is_10_dollars_l2899_289979

/-- The cost of 500 pieces of gum in dollars -/
def cost_of_500_pieces : ℚ := 10

/-- The cost of 1 piece of gum in cents -/
def cost_per_piece : ℕ := 2

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The cost of 500 pieces of gum is 10 dollars -/
theorem cost_of_500_pieces_is_10_dollars :
  cost_of_500_pieces = (500 * cost_per_piece : ℚ) / cents_per_dollar := by
  sorry

end cost_of_500_pieces_is_10_dollars_l2899_289979


namespace cube_root_8000_simplification_l2899_289990

theorem cube_root_8000_simplification :
  ∃ (a b : ℕ+), (a : ℝ) * (b : ℝ)^(1/3) = 8000^(1/3) ∧ 
  (∀ (c d : ℕ+), (c : ℝ) * (d : ℝ)^(1/3) = 8000^(1/3) → b ≤ d) ∧
  a = 20 ∧ b = 1 := by
  sorry

end cube_root_8000_simplification_l2899_289990


namespace circle_satisfies_conditions_l2899_289937

-- Define the given circles and line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 3 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 3 = 0
def line (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Define the result circle
def result_circle (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 16*y - 3 = 0

-- Theorem statement
theorem circle_satisfies_conditions :
  (∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → result_circle x y) ∧
  (∃ h k : ℝ, line h k ∧ ∀ x y : ℝ, result_circle x y ↔ (x - h)^2 + (y - k)^2 = ((h^2 + k^2 - 3) / 2)) :=
sorry

end circle_satisfies_conditions_l2899_289937


namespace probability_of_condition_l2899_289977

-- Define the bounds for x and y
def x_lower : ℝ := 0
def x_upper : ℝ := 4
def y_lower : ℝ := 0
def y_upper : ℝ := 7

-- Define the condition
def condition (x y : ℝ) : Prop := x + y ≤ 5

-- Define the probability function
def probability : ℝ := sorry

-- Theorem statement
theorem probability_of_condition : probability = 3/7 := by sorry

end probability_of_condition_l2899_289977


namespace det_A_zero_for_n_gt_five_l2899_289910

def A (n : ℕ) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j => (i.val^j.val + j.val^i.val) % 3

theorem det_A_zero_for_n_gt_five (n : ℕ) (h : n > 5) :
  Matrix.det (A n) = 0 :=
sorry

end det_A_zero_for_n_gt_five_l2899_289910


namespace sum_is_odd_l2899_289988

theorem sum_is_odd : Odd (2^1990 + 3^1990 + 7^1990 + 9^1990) := by
  sorry

end sum_is_odd_l2899_289988


namespace savings_distribution_child_receives_1680_l2899_289933

/-- Calculates the amount each child receives from a couple's savings --/
theorem savings_distribution (husband_weekly : ℕ) (wife_weekly : ℕ) 
  (months : ℕ) (weeks_per_month : ℕ) (num_children : ℕ) : ℕ :=
  let total_savings := (husband_weekly + wife_weekly) * weeks_per_month * months
  let half_savings := total_savings / 2
  half_savings / num_children

/-- Proves that each child receives $1680 given the specific conditions --/
theorem child_receives_1680 : 
  savings_distribution 335 225 6 4 4 = 1680 := by
  sorry

end savings_distribution_child_receives_1680_l2899_289933


namespace max_valid_n_l2899_289902

def S : Set ℕ := {n | ∃ x y : ℕ, n = x * y * (x + y)}

def valid (a n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (a + 2^k) ∈ S

theorem max_valid_n :
  ∃ a : ℕ, valid a 3 ∧ ∀ n : ℕ, valid a n → n ≤ 3 :=
sorry

end max_valid_n_l2899_289902


namespace max_product_sum_300_l2899_289954

theorem max_product_sum_300 : 
  ∀ a b : ℤ, a + b = 300 → a * b ≤ 22500 := by
  sorry

end max_product_sum_300_l2899_289954


namespace area_of_isosceles_right_triangle_l2899_289912

/-- Given a square ABCD and an isosceles right triangle CMN where:
  - The area of square ABCD is 4 square inches
  - MN = NC
  - x is the length of BN
Prove that the area of triangle CMN is (2 - 2x + 0.5x^2)√2 square inches -/
theorem area_of_isosceles_right_triangle (x : ℝ) :
  let abcd_area : ℝ := 4
  let cmn_is_isosceles_right : Prop := true
  let mn_eq_nc : Prop := true
  let bn_length : ℝ := x
  let cmn_area : ℝ := (2 - 2*x + 0.5*x^2) * Real.sqrt 2
  abcd_area = 4 → cmn_is_isosceles_right → mn_eq_nc → cmn_area = (2 - 2*x + 0.5*x^2) * Real.sqrt 2 :=
by
  sorry


end area_of_isosceles_right_triangle_l2899_289912


namespace point_slope_equation_l2899_289936

/-- Given a line with slope 3 passing through the point (-1, -2),
    prove that its point-slope form equation is y + 2 = 3(x + 1) -/
theorem point_slope_equation (x y : ℝ) :
  let slope : ℝ := 3
  let point : ℝ × ℝ := (-1, -2)
  (y - point.2 = slope * (x - point.1)) ↔ (y + 2 = 3 * (x + 1)) := by
sorry

end point_slope_equation_l2899_289936


namespace subset_implies_a_equals_one_l2899_289947

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

-- Theorem statement
theorem subset_implies_a_equals_one :
  ∀ a : ℝ, A a ⊆ B a → a = 1 := by
  sorry

end subset_implies_a_equals_one_l2899_289947


namespace equation_satisfies_condition_l2899_289930

theorem equation_satisfies_condition (x y z : ℤ) :
  x = y ∧ y = z + 1 → x^2 - x*y + y^2 - y*z + z^2 - z*x = 2 := by
  sorry

end equation_satisfies_condition_l2899_289930


namespace negation_of_forall_greater_than_one_l2899_289929

theorem negation_of_forall_greater_than_one (x : ℝ) : 
  ¬(∀ x > 1, x - 1 > Real.log x) ↔ ∃ x > 1, x - 1 ≤ Real.log x :=
by sorry

end negation_of_forall_greater_than_one_l2899_289929


namespace fence_cost_per_foot_l2899_289909

theorem fence_cost_per_foot
  (area : ℝ)
  (total_cost : ℝ)
  (h_area : area = 289)
  (h_total_cost : total_cost = 3944) :
  total_cost / (4 * Real.sqrt area) = 58 :=
by sorry

end fence_cost_per_foot_l2899_289909


namespace divisible_by_nine_unique_uphill_divisible_by_nine_l2899_289919

/-- An uphill integer is a positive integer where each digit is strictly greater than the previous digit. -/
def is_uphill (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

/-- A number is divisible by 9 if and only if the sum of its digits is divisible by 9. -/
theorem divisible_by_nine (n : ℕ) : n % 9 = 0 ↔ (n.digits 10).sum % 9 = 0 :=
sorry

/-- There is exactly one uphill integer divisible by 9. -/
theorem unique_uphill_divisible_by_nine : ∃! n : ℕ, is_uphill n ∧ n % 9 = 0 :=
sorry

end divisible_by_nine_unique_uphill_divisible_by_nine_l2899_289919


namespace rectangular_prism_volume_l2899_289958

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℚ := 3

/-- Length of the rectangular prism in yards -/
def length_yd : ℚ := 1

/-- Width of the rectangular prism in yards -/
def width_yd : ℚ := 2

/-- Height of the rectangular prism in yards -/
def height_yd : ℚ := 3

/-- Volume of a rectangular prism in cubic feet -/
def volume_cubic_feet (l w h : ℚ) : ℚ := l * w * h * (yards_to_feet ^ 3)

/-- Theorem stating that the volume of the given rectangular prism is 162 cubic feet -/
theorem rectangular_prism_volume :
  volume_cubic_feet length_yd width_yd height_yd = 162 := by
  sorry

end rectangular_prism_volume_l2899_289958


namespace min_value_of_expression_equality_condition_exists_l2899_289914

theorem min_value_of_expression (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem equality_condition_exists : ∃ x > 1, x + 1 / (x - 1) = 3 := by
  sorry

end min_value_of_expression_equality_condition_exists_l2899_289914


namespace simplify_and_evaluate_l2899_289956

theorem simplify_and_evaluate : 
  let a : ℝ := -2
  let b : ℝ := 1
  3 * a + 2 * (a - 1/2 * b^2) - (a - 2 * b^2) = -7 := by
  sorry

end simplify_and_evaluate_l2899_289956


namespace right_triangular_prism_volume_l2899_289941

/-- The volume of a right triangular prism with base edge length 2 and height 3 is √3. -/
theorem right_triangular_prism_volume : 
  let base_edge : ℝ := 2
  let height : ℝ := 3
  let base_area : ℝ := Real.sqrt 3 / 4 * base_edge ^ 2
  let volume : ℝ := 1 / 3 * base_area * height
  volume = Real.sqrt 3 := by
  sorry

end right_triangular_prism_volume_l2899_289941


namespace arithmetic_sequence_difference_l2899_289915

theorem arithmetic_sequence_difference (n : ℕ) (sum : ℝ) (min max : ℝ) : 
  n = 150 →
  sum = 9000 →
  min = 20 →
  max = 90 →
  let avg := sum / n
  let d := (max - min) / (2 * (n - 1))
  let L' := avg - (79 * d)
  let G' := avg + (79 * d)
  G' - L' = 6660 / 149 := by
  sorry

end arithmetic_sequence_difference_l2899_289915


namespace candidate_failing_marks_l2899_289983

def max_marks : ℕ := 500
def passing_percentage : ℚ := 45 / 100
def candidate_marks : ℕ := 180

theorem candidate_failing_marks :
  (max_marks * passing_percentage).floor - candidate_marks = 45 := by
  sorry

end candidate_failing_marks_l2899_289983


namespace value_equals_eleven_l2899_289900

theorem value_equals_eleven (number : ℝ) (value : ℝ) : 
  number = 10 → 
  (number / 2) + 6 = value → 
  value = 11 := by
sorry

end value_equals_eleven_l2899_289900


namespace arithmetic_sequence_triple_sums_l2899_289932

/-- Given an arithmetic sequence {a_n}, the sequence formed by sums of consecutive triples
    (a_1 + a_2 + a_3), (a_4 + a_5 + a_6), (a_7 + a_8 + a_9), ... is also an arithmetic sequence. -/
theorem arithmetic_sequence_triple_sums (a : ℕ → ℝ) (d : ℝ) 
    (h : ∀ n, a (n + 1) = a n + d) :
  ∃ d' : ℝ, ∀ n, (a (3*n + 1) + a (3*n + 2) + a (3*n + 3)) + d' = 
    (a (3*(n+1) + 1) + a (3*(n+1) + 2) + a (3*(n+1) + 3)) :=
by sorry

end arithmetic_sequence_triple_sums_l2899_289932
