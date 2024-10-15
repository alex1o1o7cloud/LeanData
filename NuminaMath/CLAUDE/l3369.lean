import Mathlib

namespace NUMINAMATH_CALUDE_margies_change_is_6_25_l3369_336988

/-- The amount of change Margie received after buying apples -/
def margies_change (num_apples : ℕ) (cost_per_apple : ℚ) (amount_paid : ℚ) : ℚ :=
  amount_paid - (num_apples : ℚ) * cost_per_apple

/-- Theorem stating that Margie's change is $6.25 given the problem conditions -/
theorem margies_change_is_6_25 :
  margies_change 5 (75 / 100) 10 = 25 / 4 :=
by sorry

end NUMINAMATH_CALUDE_margies_change_is_6_25_l3369_336988


namespace NUMINAMATH_CALUDE_rational_floor_equality_l3369_336980

theorem rational_floor_equality :
  ∃ (c d : ℤ), d < 100 ∧ d > 0 ∧
    ∀ k : ℕ, k ∈ Finset.range 100 → k > 0 →
      ⌊k * (c : ℚ) / d⌋ = ⌊k * (73 : ℚ) / 100⌋ := by
  sorry

end NUMINAMATH_CALUDE_rational_floor_equality_l3369_336980


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l3369_336978

theorem smallest_number_divisibility (x : ℕ) : x = 257 ↔ 
  (x > 0) ∧ 
  (∀ z : ℕ, z > 0 → z < x → ¬((z + 7) % 8 = 0 ∧ (z + 7) % 11 = 0 ∧ (z + 7) % 24 = 0)) ∧ 
  ((x + 7) % 8 = 0) ∧ 
  ((x + 7) % 11 = 0) ∧ 
  ((x + 7) % 24 = 0) := by
sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l3369_336978


namespace NUMINAMATH_CALUDE_polynomial_difference_divisibility_l3369_336906

theorem polynomial_difference_divisibility
  (p : Polynomial ℤ) (b c : ℤ) (h : b ≠ c) :
  (b - c) ∣ (p.eval b - p.eval c) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_divisibility_l3369_336906


namespace NUMINAMATH_CALUDE_min_days_team_a_is_ten_l3369_336914

/-- Represents the greening project parameters and constraints -/
structure GreeningProject where
  totalArea : ℝ
  teamARate : ℝ
  teamBRate : ℝ
  teamADailyCost : ℝ
  teamBDailyCost : ℝ
  totalBudget : ℝ

/-- Calculates the minimum number of days Team A should work -/
def minDaysTeamA (project : GreeningProject) : ℝ :=
  sorry

/-- Theorem stating the minimum number of days Team A should work -/
theorem min_days_team_a_is_ten (project : GreeningProject) :
  project.totalArea = 1800 ∧
  project.teamARate = 2 * project.teamBRate ∧
  400 / project.teamARate + 4 = 400 / project.teamBRate ∧
  project.teamADailyCost = 0.4 ∧
  project.teamBDailyCost = 0.25 ∧
  project.totalBudget = 8 →
  minDaysTeamA project = 10 := by
  sorry

#check min_days_team_a_is_ten

end NUMINAMATH_CALUDE_min_days_team_a_is_ten_l3369_336914


namespace NUMINAMATH_CALUDE_unique_n_modulo_101_l3369_336909

theorem unique_n_modulo_101 : ∃! n : ℤ, 0 ≤ n ∧ n < 101 ∧ (100 * n) % 101 = 72 % 101 ∧ n = 29 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_modulo_101_l3369_336909


namespace NUMINAMATH_CALUDE_flu_infection_rate_flu_infection_rate_proof_l3369_336932

theorem flu_infection_rate : ℝ → Prop :=
  fun x => (1 + x + x * (1 + x) = 144) → x = 11

-- The proof of the theorem
theorem flu_infection_rate_proof : flu_infection_rate 11 := by
  sorry

end NUMINAMATH_CALUDE_flu_infection_rate_flu_infection_rate_proof_l3369_336932


namespace NUMINAMATH_CALUDE_divisibility_relation_l3369_336971

theorem divisibility_relation (x y z : ℤ) (h : (11 : ℤ) ∣ (7 * x + 2 * y - 5 * z)) :
  (11 : ℤ) ∣ (3 * x - 7 * y + 12 * z) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_relation_l3369_336971


namespace NUMINAMATH_CALUDE_parabola_c_value_l3369_336945

/-- A parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 1 = -3 →   -- vertex at (-3, 1)
  p.x_coord 3 = -1 →   -- passes through (-1, 3)
  p.c = -5/2 := by
    sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3369_336945


namespace NUMINAMATH_CALUDE_barry_average_proof_l3369_336956

def barry_yards : List ℕ := [98, 107, 85, 89, 91]
def next_game_target : ℕ := 130
def total_games : ℕ := 6

theorem barry_average_proof :
  (barry_yards.sum + next_game_target) / total_games = 100 := by
  sorry

end NUMINAMATH_CALUDE_barry_average_proof_l3369_336956


namespace NUMINAMATH_CALUDE_binary_search_upper_bound_l3369_336919

theorem binary_search_upper_bound (n : ℕ) (h : n ≤ 100) :
  ∃ (k : ℕ), k ≤ 7 ∧ 2^k > n :=
sorry

end NUMINAMATH_CALUDE_binary_search_upper_bound_l3369_336919


namespace NUMINAMATH_CALUDE_double_wardrobe_with_socks_l3369_336995

/-- Represents the number of pairs of an item in a wardrobe -/
structure WardobeItem where
  pairs : Nat

/-- Represents a wardrobe with various clothing items -/
structure Wardrobe where
  socks : WardobeItem
  shoes : WardobeItem
  pants : WardobeItem
  tshirts : WardobeItem

/-- Calculates the total number of individual items in a wardrobe -/
def totalItems (w : Wardrobe) : Nat :=
  w.socks.pairs * 2 + w.shoes.pairs * 2 + w.pants.pairs + w.tshirts.pairs

/-- Theorem: Buying 35 pairs of socks doubles the number of items in Jonas' wardrobe -/
theorem double_wardrobe_with_socks (jonas : Wardrobe)
    (h1 : jonas.socks.pairs = 20)
    (h2 : jonas.shoes.pairs = 5)
    (h3 : jonas.pants.pairs = 10)
    (h4 : jonas.tshirts.pairs = 10) :
    totalItems { socks := ⟨jonas.socks.pairs + 35⟩,
                 shoes := jonas.shoes,
                 pants := jonas.pants,
                 tshirts := jonas.tshirts } = 2 * totalItems jonas := by
  sorry


end NUMINAMATH_CALUDE_double_wardrobe_with_socks_l3369_336995


namespace NUMINAMATH_CALUDE_base_conversion_equality_l3369_336924

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number in base 10 -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_equality : 
  toBase7 ((107 + 93) - 47) = [3, 0, 6] :=
sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l3369_336924


namespace NUMINAMATH_CALUDE_i_power_2016_l3369_336948

/-- The complex unit i -/
def i : ℂ := Complex.I

/-- Given properties of i -/
axiom i_power_1 : i^1 = i
axiom i_power_2 : i^2 = -1
axiom i_power_3 : i^3 = -i
axiom i_power_4 : i^4 = 1
axiom i_power_5 : i^5 = i

/-- Theorem: i^2016 = 1 -/
theorem i_power_2016 : i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_i_power_2016_l3369_336948


namespace NUMINAMATH_CALUDE_susan_age_proof_l3369_336976

def james_age_in_15_years : ℕ := 37

def james_current_age : ℕ := james_age_in_15_years - 15

def james_age_8_years_ago : ℕ := james_current_age - 8

def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2

def janet_current_age : ℕ := janet_age_8_years_ago + 8

def susan_current_age : ℕ := janet_current_age - 3

def susan_age_in_5_years : ℕ := susan_current_age + 5

theorem susan_age_proof : susan_age_in_5_years = 17 := by
  sorry

end NUMINAMATH_CALUDE_susan_age_proof_l3369_336976


namespace NUMINAMATH_CALUDE_jane_max_tickets_l3369_336901

/-- The cost of a single ticket -/
def ticket_cost : ℕ := 18

/-- Jane's available money -/
def jane_money : ℕ := 150

/-- The number of tickets required for a discount -/
def discount_threshold : ℕ := 5

/-- The discount rate as a fraction -/
def discount_rate : ℚ := 1 / 10

/-- Calculate the cost of n tickets with possible discount -/
def cost_with_discount (n : ℕ) : ℚ :=
  if n ≤ discount_threshold then
    n * ticket_cost
  else
    discount_threshold * ticket_cost + (n - discount_threshold) * ticket_cost * (1 - discount_rate)

/-- The maximum number of tickets Jane can buy -/
def max_tickets : ℕ := 8

/-- Theorem stating the maximum number of tickets Jane can buy -/
theorem jane_max_tickets :
  ∀ n : ℕ, cost_with_discount n ≤ jane_money ↔ n ≤ max_tickets :=
by sorry

end NUMINAMATH_CALUDE_jane_max_tickets_l3369_336901


namespace NUMINAMATH_CALUDE_total_shells_is_83_l3369_336928

/-- The total number of shells in the combined collection of five friends -/
def total_shells (initial_shells : ℕ) 
  (ed_limpet ed_oyster ed_conch ed_scallop : ℕ)
  (jacob_extra : ℕ)
  (marissa_limpet marissa_oyster marissa_conch marissa_scallop : ℕ)
  (priya_clam priya_mussel priya_conch priya_oyster : ℕ)
  (carlos_shells : ℕ) : ℕ :=
  initial_shells + 
  (ed_limpet + ed_oyster + ed_conch + ed_scallop) + 
  (ed_limpet + ed_oyster + ed_conch + ed_scallop + jacob_extra) +
  (marissa_limpet + marissa_oyster + marissa_conch + marissa_scallop) +
  (priya_clam + priya_mussel + priya_conch + priya_oyster) +
  carlos_shells

/-- The theorem stating that the total number of shells is 83 -/
theorem total_shells_is_83 : 
  total_shells 2 7 2 4 3 2 5 6 3 1 8 4 3 2 15 = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_is_83_l3369_336928


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3369_336968

theorem inequality_system_solution_set (x : ℝ) : 
  (Set.Icc (-2 : ℝ) 0).inter (Set.Ioo (-3 : ℝ) 1) = 
  {x | |x^2 + 5*x| < 6 ∧ |x + 1| ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3369_336968


namespace NUMINAMATH_CALUDE_orchids_planted_tomorrow_l3369_336925

/-- Proves that the number of orchid bushes to be planted tomorrow is 25 --/
theorem orchids_planted_tomorrow
  (initial : ℕ) -- Initial number of orchid bushes
  (planted_today : ℕ) -- Number of orchid bushes planted today
  (final : ℕ) -- Final number of orchid bushes
  (h1 : initial = 47)
  (h2 : planted_today = 37)
  (h3 : final = 109) :
  final - (initial + planted_today) = 25 := by
  sorry

#check orchids_planted_tomorrow

end NUMINAMATH_CALUDE_orchids_planted_tomorrow_l3369_336925


namespace NUMINAMATH_CALUDE_cost_per_pound_of_beef_l3369_336950

/-- Given a grocery bill with chicken, beef, and oil, prove the cost per pound of beef. -/
theorem cost_per_pound_of_beef
  (total_bill : ℝ)
  (chicken_weight : ℝ)
  (beef_weight : ℝ)
  (oil_volume : ℝ)
  (oil_cost : ℝ)
  (chicken_cost : ℝ)
  (h1 : total_bill = 16)
  (h2 : chicken_weight = 2)
  (h3 : beef_weight = 3)
  (h4 : oil_volume = 1)
  (h5 : oil_cost = 1)
  (h6 : chicken_cost = 3) :
  (total_bill - chicken_cost - oil_cost) / beef_weight = 4 := by
sorry

end NUMINAMATH_CALUDE_cost_per_pound_of_beef_l3369_336950


namespace NUMINAMATH_CALUDE_quadratic_solution_l3369_336921

theorem quadratic_solution (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) →
  a > b →
  a - b = 8 →
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3369_336921


namespace NUMINAMATH_CALUDE_triangle_identity_l3369_336983

/-- Definition of the △ operation for ordered pairs of real numbers -/
def triangle (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, a * d + b * c)

/-- Theorem stating that if (a, b) △ (x, y) = (a, b) for all real a and b, then (x, y) = (1, 0) -/
theorem triangle_identity (x y : ℝ) : 
  (∀ a b : ℝ, triangle a b x y = (a, b)) → (x, y) = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_identity_l3369_336983


namespace NUMINAMATH_CALUDE_tan_45_degrees_l3369_336986

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l3369_336986


namespace NUMINAMATH_CALUDE_sin_theta_value_l3369_336961

theorem sin_theta_value (θ : Real) (h : Real.cos (π / 4 - θ / 2) = 2 / 3) : 
  Real.sin θ = -1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3369_336961


namespace NUMINAMATH_CALUDE_curve_in_fourth_quadrant_implies_a_range_l3369_336967

-- Define the curve
def curve (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Theorem statement
theorem curve_in_fourth_quadrant_implies_a_range :
  (∀ x y : ℝ, curve x y a → fourth_quadrant x y) →
  a < -2 ∧ a ∈ Set.Iio (-2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_curve_in_fourth_quadrant_implies_a_range_l3369_336967


namespace NUMINAMATH_CALUDE_cos_difference_formula_l3369_336935

theorem cos_difference_formula (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2)
  (h2 : Real.cos A + Real.cos B = 1) :
  Real.cos (A - B) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_formula_l3369_336935


namespace NUMINAMATH_CALUDE_find_x_l3369_336960

theorem find_x : ∃ x : ℝ, x = 120 ∧ 5.76 = 0.12 * (0.40 * x) := by sorry

end NUMINAMATH_CALUDE_find_x_l3369_336960


namespace NUMINAMATH_CALUDE_expenditure_problem_l3369_336918

theorem expenditure_problem (initial_amount : ℝ) : 
  let remaining_after_clothes := (2/3) * initial_amount
  let remaining_after_food := (4/5) * remaining_after_clothes
  let remaining_after_travel := (3/4) * remaining_after_food
  let remaining_after_entertainment := (5/7) * remaining_after_travel
  let final_remaining := (5/6) * remaining_after_entertainment
  final_remaining = 200 → initial_amount = 840 := by
sorry

end NUMINAMATH_CALUDE_expenditure_problem_l3369_336918


namespace NUMINAMATH_CALUDE_female_rainbow_trout_count_l3369_336926

theorem female_rainbow_trout_count :
  -- Total speckled trout
  ∀ (total_speckled : ℕ),
  -- Male and female speckled trout
  ∀ (male_speckled female_speckled : ℕ),
  -- Male rainbow trout
  ∀ (male_rainbow : ℕ),
  -- Total trout
  ∀ (total_trout : ℕ),
  -- Conditions
  total_speckled = 645 →
  male_speckled = 2 * female_speckled + 45 →
  4 * male_rainbow = 3 * female_speckled →
  20 * male_rainbow = 3 * total_trout →
  -- Conclusion
  total_trout - total_speckled - male_rainbow = 205 :=
by
  sorry

end NUMINAMATH_CALUDE_female_rainbow_trout_count_l3369_336926


namespace NUMINAMATH_CALUDE_coats_collected_at_elementary_schools_l3369_336958

theorem coats_collected_at_elementary_schools 
  (total_coats : ℕ) 
  (high_school_coats : ℕ) 
  (h1 : total_coats = 9437) 
  (h2 : high_school_coats = 6922) : 
  total_coats - high_school_coats = 2515 := by
  sorry

end NUMINAMATH_CALUDE_coats_collected_at_elementary_schools_l3369_336958


namespace NUMINAMATH_CALUDE_garage_spokes_count_l3369_336966

/-- Represents a bicycle or tricycle -/
structure Vehicle where
  front_spokes : ℕ
  back_spokes : ℕ
  middle_spokes : Option ℕ

/-- The collection of vehicles in the garage -/
def garage : List Vehicle :=
  [
    { front_spokes := 12, back_spokes := 10, middle_spokes := none },
    { front_spokes := 14, back_spokes := 12, middle_spokes := none },
    { front_spokes := 10, back_spokes := 14, middle_spokes := none },
    { front_spokes := 14, back_spokes := 16, middle_spokes := some 12 }
  ]

/-- Calculates the total number of spokes for a single vehicle -/
def spokes_per_vehicle (v : Vehicle) : ℕ :=
  v.front_spokes + v.back_spokes + (v.middle_spokes.getD 0)

/-- Calculates the total number of spokes in the garage -/
def total_spokes : ℕ :=
  garage.map spokes_per_vehicle |>.sum

/-- Theorem stating that the total number of spokes in the garage is 114 -/
theorem garage_spokes_count : total_spokes = 114 := by
  sorry

end NUMINAMATH_CALUDE_garage_spokes_count_l3369_336966


namespace NUMINAMATH_CALUDE_total_chairs_l3369_336991

/-- Represents the number of chairs of each color in a classroom. -/
structure Classroom where
  blue : ℕ
  green : ℕ
  white : ℕ

/-- Defines a classroom with the given conditions. -/
def classroom : Classroom where
  blue := 10
  green := 3 * 10
  white := (3 * 10 + 10) - 13

/-- Theorem stating the total number of chairs in the classroom. -/
theorem total_chairs : classroom.blue + classroom.green + classroom.white = 67 := by
  sorry

#eval classroom.blue + classroom.green + classroom.white

end NUMINAMATH_CALUDE_total_chairs_l3369_336991


namespace NUMINAMATH_CALUDE_rod_triangle_impossibility_l3369_336985

theorem rod_triangle_impossibility (L : ℝ) (a b : ℝ) 
  (h1 : L > 0) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : a + b = L / 2) : 
  ¬(L / 2 + a > b ∧ L / 2 + b > a ∧ a + b > L / 2) := by
  sorry


end NUMINAMATH_CALUDE_rod_triangle_impossibility_l3369_336985


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3369_336911

theorem complex_fraction_simplification : 
  (((10^4+324)*(22^4+324)*(34^4+324)*(46^4+324)*(58^4+324)) / 
   ((4^4+324)*(16^4+324)*(28^4+324)*(40^4+324)*(52^4+324))) = 373 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3369_336911


namespace NUMINAMATH_CALUDE_incorrect_regression_equation_l3369_336904

-- Define the sample means
def x_mean : ℝ := 2
def y_mean : ℝ := 3

-- Define the proposed linear regression equation
def proposed_equation (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem incorrect_regression_equation :
  ¬(proposed_equation x_mean = y_mean) :=
sorry

end NUMINAMATH_CALUDE_incorrect_regression_equation_l3369_336904


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3369_336933

theorem min_value_sum_squares (x y : ℝ) (h : x + y = 4) :
  ∃ (m : ℝ), (∀ a b : ℝ, a + b = 4 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3369_336933


namespace NUMINAMATH_CALUDE_picture_area_l3369_336915

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h_frame_area : (2 * x + 5) * (y + 4) - x * y = 84) : x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l3369_336915


namespace NUMINAMATH_CALUDE_sin_180_degrees_is_zero_l3369_336927

/-- The sine of 180 degrees is 0 -/
theorem sin_180_degrees_is_zero : Real.sin (π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_180_degrees_is_zero_l3369_336927


namespace NUMINAMATH_CALUDE_max_shareholder_percentage_l3369_336975

theorem max_shareholder_percentage (n : ℕ) (k : ℕ) (p : ℚ) (h1 : n = 100) (h2 : k = 66) (h3 : p = 1/2) :
  ∀ (f : ℕ → ℚ),
    (∀ i, 0 ≤ f i) →
    (∀ i, i < n → f i ≤ 1) →
    (∀ s : Finset ℕ, s.card = k → s.sum f ≥ p) →
    (∀ i, i < n → f i ≤ 1/4) :=
sorry

end NUMINAMATH_CALUDE_max_shareholder_percentage_l3369_336975


namespace NUMINAMATH_CALUDE_fraction_of_y_l3369_336999

theorem fraction_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 10 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_y_l3369_336999


namespace NUMINAMATH_CALUDE_mixed_fruit_juice_cost_l3369_336939

/-- The cost per litre of the superfruit juice cocktail -/
def cocktail_cost : ℝ := 1399.45

/-- The cost per litre of the açaí berry juice -/
def acai_cost : ℝ := 3104.35

/-- The volume of mixed fruit juice used (in litres) -/
def mixed_fruit_volume : ℝ := 36

/-- The volume of açaí berry juice used (in litres) -/
def acai_volume : ℝ := 24

/-- The cost per litre of the mixed fruit juice -/
def mixed_fruit_cost : ℝ := 265.6166667

theorem mixed_fruit_juice_cost : 
  mixed_fruit_volume * mixed_fruit_cost + acai_volume * acai_cost = 
  cocktail_cost * (mixed_fruit_volume + acai_volume) := by
  sorry

end NUMINAMATH_CALUDE_mixed_fruit_juice_cost_l3369_336939


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l3369_336907

/-- Given three lines in the plane that intersect at two points, 
    prove that the parameter a must be either 1 or -2. -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (y₁ + 2*x₁ - 4 = 0 ∧ x₁ - y₁ + 1 = 0 ∧ a*x₁ - y₁ + 2 = 0) ∧
    (y₂ + 2*x₂ - 4 = 0 ∧ x₂ - y₂ + 1 = 0 ∧ a*x₂ - y₂ + 2 = 0) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  a = 1 ∨ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l3369_336907


namespace NUMINAMATH_CALUDE_carson_gold_stars_l3369_336944

/-- The total number of gold stars Carson earned over three days -/
def total_gold_stars (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem stating that Carson earned 26 gold stars in total -/
theorem carson_gold_stars :
  total_gold_stars 7 11 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_carson_gold_stars_l3369_336944


namespace NUMINAMATH_CALUDE_base4_addition_l3369_336974

/-- Addition in base 4 --/
def base4_add (a b c d : ℕ) : ℕ := sorry

/-- Convert a natural number to its base 4 representation --/
def to_base4 (n : ℕ) : List ℕ := sorry

theorem base4_addition :
  to_base4 (base4_add 1 13 313 1313) = [2, 0, 2, 0, 0] := by sorry

end NUMINAMATH_CALUDE_base4_addition_l3369_336974


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3369_336936

theorem sufficient_condition_for_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : a^2 + b^2 < 1) : 
  a * b + 1 > a + b := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3369_336936


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3369_336941

-- Define the rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the circle
structure Circle where
  radius : ℝ

-- Define the problem conditions
def tangent_and_midpoint (rect : Rectangle) (circ : Circle) : Prop :=
  -- Circle is tangent to sides EF and EH at their midpoints
  -- and passes through the midpoint of side FG
  True

-- Theorem statement
theorem rectangle_area_theorem (rect : Rectangle) (circ : Circle) :
  tangent_and_midpoint rect circ →
  rect.width * rect.height = 4 * circ.radius ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3369_336941


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3369_336993

/-- Given a geometric sequence {aₙ}, prove that a₃ + a₁₁ = 17 when a₇ = 4 and a₅ + a₉ = 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * r) →  -- Geometric sequence definition
  a 7 = 4 →                         -- Given condition
  a 5 + a 9 = 10 →                  -- Given condition
  a 3 + a 11 = 17 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3369_336993


namespace NUMINAMATH_CALUDE_percentage_of_sum_l3369_336962

theorem percentage_of_sum (x y : ℝ) (P : ℝ) 
  (h1 : 0.2 * (x - y) = P / 100 * (x + y))
  (h2 : y = 14.285714285714285 / 100 * x) :
  P = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sum_l3369_336962


namespace NUMINAMATH_CALUDE_marks_deposit_l3369_336923

theorem marks_deposit (mark_deposit : ℝ) (bryan_deposit : ℝ) : 
  bryan_deposit = 5 * mark_deposit - 40 →
  mark_deposit + bryan_deposit = 400 →
  mark_deposit = 400 / 6 := by
sorry

end NUMINAMATH_CALUDE_marks_deposit_l3369_336923


namespace NUMINAMATH_CALUDE_unique_prime_product_power_l3369_336979

/-- Given a natural number k, returns the product of the first k prime numbers -/
def primeProd (k : ℕ) : ℕ := sorry

/-- The only natural number k for which the product of the first k prime numbers 
    minus 1 is an exact power (greater than 1) of a natural number is 1 -/
theorem unique_prime_product_power : 
  ∀ k : ℕ, k > 0 → 
  (∃ (a n : ℕ), n > 1 ∧ primeProd k - 1 = a^n) → 
  k = 1 := by sorry

end NUMINAMATH_CALUDE_unique_prime_product_power_l3369_336979


namespace NUMINAMATH_CALUDE_linear_function_composition_function_transformation_l3369_336946

-- Part 1
def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  is_linear f → (∀ x, f (f x) = 4 * x - 1) →
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
sorry

-- Part 2
theorem function_transformation (f : ℝ → ℝ) :
  (∀ x, f (1 - x) = 2 * x^2 - x + 1) →
  (∀ x, f x = 2 * x^2 - 3 * x + 2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_composition_function_transformation_l3369_336946


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l3369_336990

-- Define the displacement function
def s (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 6 * t^2

-- Theorem: The instantaneous velocity at t = 3 is 54
theorem instantaneous_velocity_at_3 : v 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l3369_336990


namespace NUMINAMATH_CALUDE_specimen_expiration_time_l3369_336931

def seconds_in_day : ℕ := 24 * 60 * 60

def expiration_time (submission_time : Nat) (expiration_seconds : Nat) : Nat :=
  (submission_time + expiration_seconds) % seconds_in_day

theorem specimen_expiration_time :
  let submission_time : Nat := 15 * 60 * 60  -- 3 PM in seconds
  let expiration_seconds : Nat := 7 * 6 * 5 * 4 * 3 * 2 * 1  -- 7!
  expiration_time submission_time expiration_seconds = 16 * 60 * 60 + 24 * 60  -- 4:24 PM in seconds
  := by sorry

end NUMINAMATH_CALUDE_specimen_expiration_time_l3369_336931


namespace NUMINAMATH_CALUDE_sequence_sum_l3369_336908

theorem sequence_sum (n : ℕ) (S_n : ℝ) (a : ℕ → ℝ) : 
  (∀ k ≥ 1, a k = 1 / (Real.sqrt (k + 1) + Real.sqrt k)) →
  S_n = Real.sqrt 101 - 1 →
  n = 100 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3369_336908


namespace NUMINAMATH_CALUDE_book_pages_l3369_336965

/-- The number of pages Mrs. Hilt read -/
def pages_read : ℕ := 11

/-- The number of pages Mrs. Hilt has left to read -/
def pages_left : ℕ := 6

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_read + pages_left

theorem book_pages : total_pages = 17 := by sorry

end NUMINAMATH_CALUDE_book_pages_l3369_336965


namespace NUMINAMATH_CALUDE_tobias_change_l3369_336920

def shoe_cost : ℕ := 95
def saving_months : ℕ := 3
def monthly_allowance : ℕ := 5
def lawn_mowing_charge : ℕ := 15
def driveway_shoveling_charge : ℕ := 7
def lawns_mowed : ℕ := 4
def driveways_shoveled : ℕ := 5

def total_savings : ℕ := 
  saving_months * monthly_allowance + 
  lawns_mowed * lawn_mowing_charge + 
  driveways_shoveled * driveway_shoveling_charge

theorem tobias_change : total_savings - shoe_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_tobias_change_l3369_336920


namespace NUMINAMATH_CALUDE_dart_board_partitions_l3369_336969

def partition_count (n : ℕ) (k : ℕ) : ℕ := 
  sorry

theorem dart_board_partitions : partition_count 5 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dart_board_partitions_l3369_336969


namespace NUMINAMATH_CALUDE_perpendicular_slope_l3369_336930

/-- Given a line with equation 4x - 5y = 10, the slope of the perpendicular line is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 10) → (slope_of_perpendicular_line = -5/4) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l3369_336930


namespace NUMINAMATH_CALUDE_subset_intersection_complement_empty_l3369_336934

theorem subset_intersection_complement_empty
  {U : Type} [Nonempty U]
  (M N : Set U)
  (h : M ⊆ N) :
  M ∩ (Set.univ \ N) = ∅ := by
sorry

end NUMINAMATH_CALUDE_subset_intersection_complement_empty_l3369_336934


namespace NUMINAMATH_CALUDE_bat_wings_area_is_three_and_half_l3369_336955

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  /-- The width of the rectangle -/
  width : ℝ
  /-- The height of the rectangle -/
  height : ℝ
  /-- The length of segments DC, CB, and BA -/
  segment_length : ℝ
  /-- Width is 3 -/
  width_is_three : width = 3
  /-- Height is 4 -/
  height_is_four : height = 4
  /-- Segment length is 1 -/
  segment_is_one : segment_length = 1

/-- The area of the "bat wings" in the special rectangle -/
def batWingsArea (r : SpecialRectangle) : ℝ := sorry

/-- Theorem stating that the area of the "bat wings" is 3 1/2 -/
theorem bat_wings_area_is_three_and_half (r : SpecialRectangle) :
  batWingsArea r = 3.5 := by sorry

end NUMINAMATH_CALUDE_bat_wings_area_is_three_and_half_l3369_336955


namespace NUMINAMATH_CALUDE_reciprocal_of_difference_l3369_336949

theorem reciprocal_of_difference : (((1 : ℚ) / 3 - (1 : ℚ) / 4)⁻¹ : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_difference_l3369_336949


namespace NUMINAMATH_CALUDE_quadratic_function_value_l3369_336951

/-- A quadratic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - m * x + 5

/-- The derivative of f with respect to x -/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 4 * x - m

theorem quadratic_function_value (m : ℝ) :
  (∀ x ≥ -2, f_deriv m x ≥ 0) → f m 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l3369_336951


namespace NUMINAMATH_CALUDE_algorithm_output_l3369_336987

def algorithm (x y : Int) : (Int × Int) :=
  let x' := if x < 0 then y + 3 else x
  let y' := if x < 0 then y else y - 3
  (x' - y', y' + x')

theorem algorithm_output : algorithm (-5) 15 = (3, 33) := by
  sorry

end NUMINAMATH_CALUDE_algorithm_output_l3369_336987


namespace NUMINAMATH_CALUDE_factorial_simplification_l3369_336905

theorem factorial_simplification : (12 : ℕ).factorial / ((10 : ℕ).factorial + 3 * (9 : ℕ).factorial) = 1320 / 13 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l3369_336905


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3369_336957

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - 4|

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2 ∨ x > Real.sqrt 2}

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, f (x^2 + 2) > f x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3369_336957


namespace NUMINAMATH_CALUDE_divide_ten_theorem_l3369_336963

theorem divide_ten_theorem (x : ℝ) : 
  x > 0 ∧ x < 10 →
  (10 - x)^2 + x^2 + (10 - x) / x = 72 →
  x = 2 := by
sorry


end NUMINAMATH_CALUDE_divide_ten_theorem_l3369_336963


namespace NUMINAMATH_CALUDE_annulus_area_l3369_336903

/-- The area of an annulus formed by two concentric circles. -/
theorem annulus_area (r s x : ℝ) (hr : r > 0) (hs : s > 0) (hrs : r > s) :
  let P := Real.sqrt (r^2 - s^2)
  x^2 = r^2 - s^2 →
  π * (r^2 - s^2) = π * x^2 := by sorry

end NUMINAMATH_CALUDE_annulus_area_l3369_336903


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_value_l3369_336981

theorem square_plus_inverse_square_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_value_l3369_336981


namespace NUMINAMATH_CALUDE_product_of_solutions_l3369_336940

theorem product_of_solutions : ∃ (x₁ x₂ : ℝ), 
  (|5 * x₁| + 4 = 44) ∧ 
  (|5 * x₂| + 4 = 44) ∧ 
  (x₁ ≠ x₂) ∧
  (x₁ * x₂ = -64) := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3369_336940


namespace NUMINAMATH_CALUDE_final_elevation_proof_l3369_336902

def elevation_problem (start_elevation : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  start_elevation - rate * time

theorem final_elevation_proof (start_elevation : ℝ) (rate : ℝ) (time : ℝ)
  (h1 : start_elevation = 400)
  (h2 : rate = 10)
  (h3 : time = 5) :
  elevation_problem start_elevation rate time = 350 := by
  sorry

end NUMINAMATH_CALUDE_final_elevation_proof_l3369_336902


namespace NUMINAMATH_CALUDE_difference_of_two_numbers_l3369_336942

theorem difference_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (product_eq : x * y = 200) : 
  |x - y| = 10 := by
sorry

end NUMINAMATH_CALUDE_difference_of_two_numbers_l3369_336942


namespace NUMINAMATH_CALUDE_submarine_age_conversion_l3369_336977

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ × ℕ × ℕ) : ℕ :=
  let (a, b, c) := octal
  a * 8^2 + b * 8^1 + c * 8^0

theorem submarine_age_conversion :
  octal_to_decimal (3, 6, 7) = 247 := by
  sorry

end NUMINAMATH_CALUDE_submarine_age_conversion_l3369_336977


namespace NUMINAMATH_CALUDE_integer_distance_implies_horizontal_segment_l3369_336998

/-- A polynomial function with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- The squared Euclidean distance between two points -/
def squaredDistance (x₁ y₁ x₂ y₂ : ℤ) : ℤ :=
  (x₂ - x₁)^2 + (y₂ - y₁)^2

theorem integer_distance_implies_horizontal_segment
  (f : IntPolynomial) (a b : ℤ) :
  (∃ d : ℤ, d^2 = squaredDistance a (f a) b (f b)) →
  f a = f b :=
sorry

end NUMINAMATH_CALUDE_integer_distance_implies_horizontal_segment_l3369_336998


namespace NUMINAMATH_CALUDE_exponential_curve_logarithm_relation_l3369_336929

/-- Proves the relationship between u, b, x, and c for an exponential curve -/
theorem exponential_curve_logarithm_relation 
  (a b x : ℝ) 
  (y : ℝ := a * Real.exp (b * x)) 
  (u : ℝ := Real.log y) 
  (c : ℝ := Real.log a) : 
  u = b * x + c := by
  sorry

end NUMINAMATH_CALUDE_exponential_curve_logarithm_relation_l3369_336929


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3369_336938

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the tangent condition
def is_tangent (x y : ℝ) : Prop := ∃ (t : ℝ), circle_M (x + t) (y + t) ∧ 
  ∀ (s : ℝ), s ≠ t → ¬(circle_M (x + s) (y + s))

-- Define the minimization condition
def is_minimized (P M A B : ℝ × ℝ) : Prop := 
  ∀ (Q : ℝ × ℝ), point_P Q.1 Q.2 → 
    (Q.1 - M.1)^2 + (Q.2 - M.2)^2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥
    (P.1 - M.1)^2 + (P.2 - M.2)^2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem tangent_line_equation :
  ∀ (P M A B : ℝ × ℝ),
    circle_M M.1 M.2 →
    point_P P.1 P.2 →
    is_tangent (P.1 - A.1) (P.2 - A.2) →
    is_tangent (P.1 - B.1) (P.2 - B.2) →
    is_minimized P M A B →
    2 * A.1 + A.2 + 1 = 0 ∧ 2 * B.1 + B.2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3369_336938


namespace NUMINAMATH_CALUDE_lower_right_is_one_l3369_336982

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Check if a number appears exactly once in each row --/
def valid_rows (g : Grid) : Prop :=
  ∀ i : Fin 5, ∀ n : Fin 5, (∃! j : Fin 5, g i j = n)

/-- Check if a number appears exactly once in each column --/
def valid_columns (g : Grid) : Prop :=
  ∀ j : Fin 5, ∀ n : Fin 5, (∃! i : Fin 5, g i j = n)

/-- Check if no number repeats on the main diagonal --/
def valid_diagonal (g : Grid) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i i ≠ g j j

/-- Check if the grid satisfies the initial placements --/
def valid_initial (g : Grid) : Prop :=
  g 0 0 = 0 ∧ g 1 1 = 1 ∧ g 2 2 = 2 ∧ g 0 3 = 3 ∧ g 3 3 = 4 ∧ g 1 4 = 0

theorem lower_right_is_one (g : Grid) 
  (h_rows : valid_rows g) 
  (h_cols : valid_columns g) 
  (h_diag : valid_diagonal g) 
  (h_init : valid_initial g) : 
  g 4 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_lower_right_is_one_l3369_336982


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3369_336954

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, x^3 - y^3 = 2*x*y + 8 ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3369_336954


namespace NUMINAMATH_CALUDE_kaleb_shirts_l3369_336953

theorem kaleb_shirts (initial_shirts : ℕ) (removed_shirts : ℕ) :
  initial_shirts = 17 →
  removed_shirts = 7 →
  initial_shirts - removed_shirts = 10 :=
by sorry

end NUMINAMATH_CALUDE_kaleb_shirts_l3369_336953


namespace NUMINAMATH_CALUDE_cindy_marbles_l3369_336997

/-- Given Cindy's initial marbles and distribution to friends, calculate five times her remaining marbles -/
theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 800)
  (h2 : friends = 6)
  (h3 : marbles_per_friend = 120) :
  5 * (initial_marbles - friends * marbles_per_friend) = 400 := by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_l3369_336997


namespace NUMINAMATH_CALUDE_solve_system_l3369_336917

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 18) : y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3369_336917


namespace NUMINAMATH_CALUDE_lemonade_water_calculation_l3369_336984

/-- Represents the ratio of water to lemon juice in the lemonade recipe -/
def water_to_juice_ratio : ℚ := 5 / 3

/-- Represents the number of gallons of lemonade to be made -/
def gallons_to_make : ℚ := 2

/-- Represents the number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- Calculates the number of quarts of water needed for the lemonade recipe -/
def quarts_of_water_needed : ℚ :=
  (water_to_juice_ratio * gallons_to_make * quarts_per_gallon) / (water_to_juice_ratio + 1)

/-- Theorem stating that 5 quarts of water are needed for the lemonade recipe -/
theorem lemonade_water_calculation :
  quarts_of_water_needed = 5 := by sorry

end NUMINAMATH_CALUDE_lemonade_water_calculation_l3369_336984


namespace NUMINAMATH_CALUDE_horse_lap_time_l3369_336992

-- Define the given parameters
def field_area : Real := 625
def horse_speed : Real := 25

-- Define the theorem
theorem horse_lap_time : 
  ∀ (side_length perimeter time : Real),
  side_length^2 = field_area →
  perimeter = 4 * side_length →
  time = perimeter / horse_speed →
  time = 4 := by
  sorry

end NUMINAMATH_CALUDE_horse_lap_time_l3369_336992


namespace NUMINAMATH_CALUDE_circle_area_increase_l3369_336973

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l3369_336973


namespace NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l3369_336916

/-- The shortest distance from a circle to a line --/
theorem shortest_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | (x - 3)^2 + (y + 3)^2 = 9}
  let line := {(x, y) : ℝ × ℝ | y = x}
  (∃ (d : ℝ), d = 3 * (Real.sqrt 2 - 1) ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      d ≤ Real.sqrt ((p.1 - p.2)^2 / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l3369_336916


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l3369_336964

theorem correct_quotient_proof (dividend : ℕ) (mistaken_divisor correct_divisor mistaken_quotient : ℕ) 
  (h1 : mistaken_divisor = 12)
  (h2 : correct_divisor = 21)
  (h3 : mistaken_quotient = 42)
  (h4 : dividend = mistaken_divisor * mistaken_quotient)
  (h5 : dividend % correct_divisor = 0) :
  dividend / correct_divisor = 24 := by
sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l3369_336964


namespace NUMINAMATH_CALUDE_cube_root_seven_to_sixth_l3369_336913

theorem cube_root_seven_to_sixth (x : ℝ) : x = 7^(1/3) → x^6 = 49 := by sorry

end NUMINAMATH_CALUDE_cube_root_seven_to_sixth_l3369_336913


namespace NUMINAMATH_CALUDE_combinatorial_sum_equality_l3369_336989

theorem combinatorial_sum_equality (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  (Finset.range (k + 1)).sum (λ j => Nat.choose k j * Nat.choose n (m - j)) = Nat.choose (n + k) m := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_sum_equality_l3369_336989


namespace NUMINAMATH_CALUDE_fraction_addition_l3369_336952

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3369_336952


namespace NUMINAMATH_CALUDE_equation_solution_l3369_336994

theorem equation_solution : ∃ x : ℝ, 11 + Real.sqrt (x + 6 * 4 / 3) = 13 ∧ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3369_336994


namespace NUMINAMATH_CALUDE_binomial_divisibility_l3369_336900

theorem binomial_divisibility (n k : ℕ) (h_k : k > 1) :
  (∀ m : ℕ, 1 ≤ m ∧ m < n → k ∣ Nat.choose n m) ↔
  ∃ (p : ℕ) (t : ℕ+), Nat.Prime p ∧ n = p ^ (t : ℕ) ∧ k = p :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l3369_336900


namespace NUMINAMATH_CALUDE_regular_polygon_with_60_degree_exterior_angle_has_6_sides_l3369_336910

/-- The number of sides of a regular polygon with an exterior angle of 60 degrees is 6. -/
theorem regular_polygon_with_60_degree_exterior_angle_has_6_sides :
  ∀ (n : ℕ), n > 0 →
  (360 / n = 60) →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_60_degree_exterior_angle_has_6_sides_l3369_336910


namespace NUMINAMATH_CALUDE_vector_sum_proof_l3369_336922

/-- Given vectors a and b in ℝ², prove that a + 2b = (-3, 4) -/
theorem vector_sum_proof (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-2, 1)) :
  a + 2 • b = (-3, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l3369_336922


namespace NUMINAMATH_CALUDE_min_value_on_interval_l3369_336996

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = -15 ∧ ∀ y ∈ Set.Icc 0 3, f y ≥ f x := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l3369_336996


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3369_336937

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, (1202 + m) % 4 = 0 → n ≤ m) ∧ (1202 + n) % 4 = 0 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3369_336937


namespace NUMINAMATH_CALUDE_x_makes_2n_plus_x_composite_x_is_correct_l3369_336947

/-- The number added to 2n to make it not prime when n = 4 -/
def x : ℕ := 1

/-- The smallest n for which 2n + x is not prime -/
def smallest_n : ℕ := 4

theorem x_makes_2n_plus_x_composite : 
  ¬ Nat.Prime (2 * smallest_n + x) ∧ 
  ∀ m < smallest_n, Nat.Prime (2 * m + x) := by
  sorry

theorem x_is_correct : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_makes_2n_plus_x_composite_x_is_correct_l3369_336947


namespace NUMINAMATH_CALUDE_arithmetic_expression_proof_l3369_336959

/-- Proves that the given arithmetic expression evaluates to 1320 -/
theorem arithmetic_expression_proof : 1583 + 240 / 60 * 5 - 283 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_proof_l3369_336959


namespace NUMINAMATH_CALUDE_tangent_lines_count_l3369_336912

theorem tangent_lines_count : ∃! (s : Finset ℝ), 
  (∀ x₀ ∈ s, x₀ * Real.exp x₀ * (x₀^2 - x₀ - 4) = 0) ∧ 
  Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l3369_336912


namespace NUMINAMATH_CALUDE_highest_probability_red_card_l3369_336943

theorem highest_probability_red_card (total_cards : Nat) (ace_cards : Nat) (heart_cards : Nat) (king_cards : Nat) (red_cards : Nat) :
  total_cards = 52 →
  ace_cards = 4 →
  heart_cards = 13 →
  king_cards = 4 →
  red_cards = 26 →
  (red_cards : ℚ) / total_cards > (heart_cards : ℚ) / total_cards ∧
  (red_cards : ℚ) / total_cards > (ace_cards : ℚ) / total_cards ∧
  (red_cards : ℚ) / total_cards > (king_cards : ℚ) / total_cards :=
by sorry

end NUMINAMATH_CALUDE_highest_probability_red_card_l3369_336943


namespace NUMINAMATH_CALUDE_sara_quarters_remaining_l3369_336972

/-- Given that Sara initially had 783 quarters and her dad borrowed 271 quarters,
    prove that Sara now has 512 quarters. -/
theorem sara_quarters_remaining (initial_quarters borrowed_quarters : ℕ) 
    (h1 : initial_quarters = 783)
    (h2 : borrowed_quarters = 271) :
    initial_quarters - borrowed_quarters = 512 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_remaining_l3369_336972


namespace NUMINAMATH_CALUDE_short_track_speed_skating_selection_l3369_336970

theorem short_track_speed_skating_selection
  (p : Prop) -- A gets first place
  (q : Prop) -- B gets second place
  (r : Prop) -- C gets third place
  (h1 : p ∨ q) -- p ∨ q is true
  (h2 : ¬(p ∧ q)) -- p ∧ q is false
  (h3 : (¬q) ∧ r) -- (¬q) ∧ r is true
  : p ∧ ¬q ∧ r := by sorry

end NUMINAMATH_CALUDE_short_track_speed_skating_selection_l3369_336970
