import Mathlib

namespace product_sequence_sum_l1216_121618

theorem product_sequence_sum (c d : ℕ) (h1 : c / 3 = 12) (h2 : d = c - 1) : c + d = 71 := by
  sorry

end product_sequence_sum_l1216_121618


namespace atomic_mass_scientific_notation_l1216_121633

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem atomic_mass_scientific_notation :
  toScientificNotation 0.00001992 = ScientificNotation.mk 1.992 (-5) sorry := by
  sorry

end atomic_mass_scientific_notation_l1216_121633


namespace quadratic_equation_results_l1216_121670

theorem quadratic_equation_results (y : ℝ) (h : 6 * y^2 + 7 = 5 * y + 12) : 
  ((12 * y - 5)^2 = 145) ∧ 
  ((5 * y + 2)^2 = (4801 + 490 * Real.sqrt 145 + 3625) / 144 ∨
   (5 * y + 2)^2 = (4801 - 490 * Real.sqrt 145 + 3625) / 144) := by
  sorry

end quadratic_equation_results_l1216_121670


namespace regular_polygon_140_degree_interior_l1216_121631

/-- A regular polygon with interior angles measuring 140° has 9 sides. -/
theorem regular_polygon_140_degree_interior : ∀ n : ℕ, 
  n > 2 → -- ensure it's a valid polygon
  (180 * (n - 2) : ℝ) = (140 * n : ℝ) → -- sum of interior angles formula
  n = 9 := by
  sorry

end regular_polygon_140_degree_interior_l1216_121631


namespace triangle_properties_l1216_121667

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a = 2 * Real.sqrt 3 →
  A = π / 3 →
  12 = b^2 + c^2 - b*c →
  (∃ (S : ℝ), S = (Real.sqrt 3 / 4) * b * c ∧ S ≤ 3 * Real.sqrt 3 ∧
    (S = 3 * Real.sqrt 3 → b = c)) ∧
  (a + b + c ≤ 6 * Real.sqrt 3 ∧
    (a + b + c = 6 * Real.sqrt 3 → b = c)) ∧
  (0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
    1/2 < b/c ∧ b/c < 2) :=
by sorry

end triangle_properties_l1216_121667


namespace football_hits_ground_time_l1216_121690

def football_height (t : ℝ) : ℝ := -16 * t^2 + 18 * t + 60

theorem football_hits_ground_time :
  ∃ t : ℝ, t > 0 ∧ football_height t = 0 ∧ t = 41 / 16 := by
  sorry

end football_hits_ground_time_l1216_121690


namespace triangle_max_area_l1216_121687

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area is √3 when (a+b)(sin A - sin B) = (c-b)sin C and a = 2 -/
theorem triangle_max_area (a b c A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  ((a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) →
  (a = 2) →
  (∃ (S : ℝ), S ≤ Real.sqrt 3 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * b * c * Real.sin A → S' ≤ S) :=
by sorry

end triangle_max_area_l1216_121687


namespace alcohol_solution_percentage_l1216_121624

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) :
  initial_volume = 6 →
  initial_percentage = 0.3 →
  added_alcohol = 2.4 →
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_percentage := final_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end alcohol_solution_percentage_l1216_121624


namespace inequality_proof_l1216_121612

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 * (y^2 + z^2)^2 + y^3 * (z^2 + x^2)^2 + z^3 * (x^2 + y^2)^2 ≥
  x * y * z * (x * y * (x + y)^2 + y * z * (y + z)^2 + z * x * (z + x)^2) := by
  sorry

end inequality_proof_l1216_121612


namespace smaller_rectangle_area_percentage_l1216_121686

/-- A circle with a rectangle inscribed in it -/
structure InscribedRectangle where
  center : ℝ × ℝ
  radius : ℝ
  rect_length : ℝ
  rect_width : ℝ

/-- A smaller rectangle with one side coinciding with the larger rectangle and two vertices on the circle -/
structure SmallerRectangle where
  length : ℝ
  width : ℝ

/-- The configuration of the inscribed rectangle and the smaller rectangle -/
structure Configuration where
  inscribed : InscribedRectangle
  smaller : SmallerRectangle

/-- The theorem stating that the area of the smaller rectangle is 0% of the area of the larger rectangle -/
theorem smaller_rectangle_area_percentage (config : Configuration) : 
  (config.smaller.length * config.smaller.width) / (config.inscribed.rect_length * config.inscribed.rect_width) = 0 := by
  sorry

end smaller_rectangle_area_percentage_l1216_121686


namespace only_sqrt_8_not_simplest_l1216_121607

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Define a function to check if a radical is in its simplest form
def isSimplestForm (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → ¬isPerfectSquare m

-- Theorem statement
theorem only_sqrt_8_not_simplest : 
  isSimplestForm 10 ∧ 
  isSimplestForm 6 ∧ 
  isSimplestForm 2 ∧ 
  ¬isSimplestForm 8 :=
sorry

end only_sqrt_8_not_simplest_l1216_121607


namespace min_posts_for_fence_l1216_121698

def fence_length : ℝ := 40 + 40 + 100
def post_spacing : ℝ := 10
def area_width : ℝ := 40
def area_length : ℝ := 100

theorem min_posts_for_fence : 
  ⌊fence_length / post_spacing⌋ + 1 = 19 := by sorry

end min_posts_for_fence_l1216_121698


namespace trig_sum_simplification_l1216_121605

theorem trig_sum_simplification :
  (Real.sin (30 * π / 180) + Real.sin (50 * π / 180) + Real.sin (70 * π / 180) + Real.sin (90 * π / 180) +
   Real.sin (110 * π / 180) + Real.sin (130 * π / 180) + Real.sin (150 * π / 180) + Real.sin (170 * π / 180)) /
  (Real.cos (15 * π / 180) * Real.cos (25 * π / 180) * Real.cos (50 * π / 180)) =
  (8 * Real.sin (80 * π / 180) * Real.cos (40 * π / 180) * Real.cos (20 * π / 180)) /
  (Real.cos (15 * π / 180) * Real.cos (25 * π / 180) * Real.cos (50 * π / 180)) :=
by
  sorry

end trig_sum_simplification_l1216_121605


namespace band_gigs_played_l1216_121637

/-- Represents the earnings of each band member per gig -/
structure BandEarnings :=
  (leadSinger : ℕ)
  (guitarist : ℕ)
  (bassist : ℕ)
  (drummer : ℕ)
  (keyboardist : ℕ)
  (backupSinger1 : ℕ)
  (backupSinger2 : ℕ)
  (backupSinger3 : ℕ)

/-- Calculates the total earnings per gig for the band -/
def totalEarningsPerGig (earnings : BandEarnings) : ℕ :=
  earnings.leadSinger + earnings.guitarist + earnings.bassist + earnings.drummer +
  earnings.keyboardist + earnings.backupSinger1 + earnings.backupSinger2 + earnings.backupSinger3

/-- Theorem: The band has played 21 gigs -/
theorem band_gigs_played (earnings : BandEarnings) 
  (h1 : earnings.leadSinger = 30)
  (h2 : earnings.guitarist = 25)
  (h3 : earnings.bassist = 20)
  (h4 : earnings.drummer = 25)
  (h5 : earnings.keyboardist = 20)
  (h6 : earnings.backupSinger1 = 15)
  (h7 : earnings.backupSinger2 = 18)
  (h8 : earnings.backupSinger3 = 12)
  (h9 : totalEarningsPerGig earnings * 21 = 3465) :
  21 = 3465 / (totalEarningsPerGig earnings) :=
by sorry

end band_gigs_played_l1216_121637


namespace unique_solution_for_2n_plus_m_l1216_121668

theorem unique_solution_for_2n_plus_m : 
  ∀ n m : ℤ, 
    (3 * n - m < 5) → 
    (n + m > 26) → 
    (3 * m - 2 * n < 46) → 
    (2 * n + m = 36) :=
by
  sorry

end unique_solution_for_2n_plus_m_l1216_121668


namespace inequality_proof_l1216_121695

/-- For all real x greater than -1, 1 - e^(-x) is greater than or equal to x/(x+1) -/
theorem inequality_proof (x : ℝ) (h : x > -1) : 1 - Real.exp (-x) ≥ x / (x + 1) := by
  sorry

end inequality_proof_l1216_121695


namespace line_translation_theorem_l1216_121602

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically -/
def translate_line (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + units }

theorem line_translation_theorem (original : Line) (units : ℝ) :
  original.slope = 1/2 ∧ original.intercept = -2 ∧ units = 3 →
  translate_line original units = Line.mk (1/2) 1 :=
by sorry

end line_translation_theorem_l1216_121602


namespace expression_value_l1216_121627

/-- Given that px³ + qx + 3 = 2005 when x = 3, prove that px³ + qx + 3 = -1999 when x = -3 -/
theorem expression_value (p q : ℝ) : 
  (27 * p + 3 * q + 3 = 2005) → (-27 * p - 3 * q + 3 = -1999) := by
sorry

end expression_value_l1216_121627


namespace amp_six_three_l1216_121661

/-- The & operation defined on two real numbers -/
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that 6 & 3 = 27 -/
theorem amp_six_three : amp 6 3 = 27 := by
  sorry

end amp_six_three_l1216_121661


namespace min_value_theorem_l1216_121650

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_line : 6 * a + 3 * b = 1) :
  1 / (5 * a + 2 * b) + 2 / (a + b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_value_theorem_l1216_121650


namespace student_average_less_than_true_average_l1216_121675

theorem student_average_less_than_true_average 
  (x y w : ℝ) (h : x > y ∧ y > w) : 
  ((x + y) / 2 + w) / 2 < (x + y + w) / 3 := by
sorry

end student_average_less_than_true_average_l1216_121675


namespace fraction_simplification_l1216_121621

theorem fraction_simplification : (8 + 4) / (8 - 4) = 3 := by sorry

end fraction_simplification_l1216_121621


namespace yeast_growth_proof_l1216_121638

def yeast_population (initial_population : ℕ) (growth_factor : ℕ) (interval : ℕ) (time : ℕ) : ℕ :=
  initial_population * growth_factor ^ (time / interval)

theorem yeast_growth_proof (initial_population : ℕ) (growth_factor : ℕ) (interval : ℕ) (time : ℕ) :
  initial_population = 50 →
  growth_factor = 3 →
  interval = 5 →
  time = 20 →
  yeast_population initial_population growth_factor interval time = 4050 :=
by
  sorry

end yeast_growth_proof_l1216_121638


namespace least_integer_with_digit_sum_property_l1216_121609

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: 2999999999999 is the least positive integer N such that
    the sum of its digits is 100 and the sum of the digits of 2N is 110 -/
theorem least_integer_with_digit_sum_property : 
  (∀ m : ℕ, m > 0 ∧ m < 2999999999999 → 
    (sum_of_digits m = 100 ∧ sum_of_digits (2 * m) = 110) → False) ∧ 
  sum_of_digits 2999999999999 = 100 ∧ 
  sum_of_digits (2 * 2999999999999) = 110 :=
sorry

end least_integer_with_digit_sum_property_l1216_121609


namespace tessa_final_debt_l1216_121689

/-- Calculates the final debt given an initial debt, a fractional repayment, and an additional loan --/
def finalDebt (initialDebt : ℚ) (repaymentFraction : ℚ) (additionalLoan : ℚ) : ℚ :=
  initialDebt - (repaymentFraction * initialDebt) + additionalLoan

/-- Proves that Tessa's final debt is $30 --/
theorem tessa_final_debt :
  finalDebt 40 (1/2) 10 = 30 := by
  sorry

end tessa_final_debt_l1216_121689


namespace pauls_weekly_spending_l1216_121646

/-- Given Paul's earnings and the duration the money lasted, calculate his weekly spending. -/
theorem pauls_weekly_spending (lawn_mowing : ℕ) (weed_eating : ℕ) (weeks : ℕ) 
  (h1 : lawn_mowing = 44)
  (h2 : weed_eating = 28)
  (h3 : weeks = 8)
  (h4 : weeks > 0) :
  (lawn_mowing + weed_eating) / weeks = 9 := by
  sorry

#check pauls_weekly_spending

end pauls_weekly_spending_l1216_121646


namespace total_pages_calculation_l1216_121623

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 12

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 75

/-- The total number of pages in all booklets -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem total_pages_calculation : total_pages = 900 := by
  sorry

end total_pages_calculation_l1216_121623


namespace factorial_simplification_l1216_121678

theorem factorial_simplification :
  (13 : ℕ).factorial / ((11 : ℕ).factorial + 3 * (9 : ℕ).factorial) = 17160 / 113 := by
  sorry

end factorial_simplification_l1216_121678


namespace city_population_multiple_l1216_121665

/- Define the populations of the cities and the multiple -/
def willowdale_population : ℕ := 2000
def sun_city_population : ℕ := 12000

/- Define the relationship between the cities' populations -/
def roseville_population (m : ℕ) : ℤ := m * willowdale_population - 500
def sun_city_relation (m : ℕ) : Prop := 
  sun_city_population = 2 * (roseville_population m) + 1000

/- State the theorem -/
theorem city_population_multiple : ∃ m : ℕ, sun_city_relation m ∧ m = 3 := by
  sorry

end city_population_multiple_l1216_121665


namespace pentagon_angle_sum_l1216_121663

-- Define the pentagon and its angles
structure Pentagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  G : ℝ

-- Define the theorem
theorem pentagon_angle_sum (p : Pentagon) 
  (h1 : p.A = 40)
  (h2 : p.F = p.G) : 
  p.B + p.D = 70 := by
  sorry

#check pentagon_angle_sum

end pentagon_angle_sum_l1216_121663


namespace task_completion_correct_l1216_121683

/-- Represents the number of days it takes for a person to complete the task alone -/
structure PersonWorkRate where
  days : ℝ
  days_positive : days > 0

/-- Represents the scenario of two people working on a task -/
structure WorkScenario where
  person_a : PersonWorkRate
  person_b : PersonWorkRate
  days_a_alone : ℝ
  days_together : ℝ
  days_a_alone_nonnegative : days_a_alone ≥ 0
  days_together_nonnegative : days_together ≥ 0

/-- The equation representing the completion of the task -/
def task_completion_equation (scenario : WorkScenario) : Prop :=
  (scenario.days_together + scenario.days_a_alone) / scenario.person_a.days +
  scenario.days_together / scenario.person_b.days = 1

/-- The theorem stating that the given equation correctly represents the completion of the task -/
theorem task_completion_correct (scenario : WorkScenario)
  (h1 : scenario.person_a.days = 3)
  (h2 : scenario.person_b.days = 5)
  (h3 : scenario.days_a_alone = 1) :
  task_completion_equation scenario :=
sorry

end task_completion_correct_l1216_121683


namespace min_force_to_prevent_slipping_l1216_121632

/-- The minimum force needed to keep a book from slipping -/
theorem min_force_to_prevent_slipping 
  (M : ℝ) -- Mass of the book
  (g : ℝ) -- Acceleration due to gravity
  (μs : ℝ) -- Coefficient of static friction
  (h1 : M > 0) -- Mass is positive
  (h2 : g > 0) -- Gravity is positive
  (h3 : μs > 0) -- Coefficient of static friction is positive
  : 
  ∃ (F : ℝ), F = M * g / μs ∧ F ≥ M * g ∧ ∀ (F' : ℝ), F' < F → F' * μs < M * g :=
sorry

end min_force_to_prevent_slipping_l1216_121632


namespace eighth_of_2_38_l1216_121696

theorem eighth_of_2_38 (x : ℕ) :
  (1 / 8 : ℝ) * (2 : ℝ)^38 = (2 : ℝ)^x → x = 35 := by
  sorry

end eighth_of_2_38_l1216_121696


namespace min_vegetable_dishes_l1216_121692

theorem min_vegetable_dishes (n : ℕ) (h : n ≥ 5) :
  (∃ x : ℕ, x ≥ 7 ∧ Nat.choose n 2 * Nat.choose x 2 > 200) ∧
  (∀ y : ℕ, y < 7 → Nat.choose n 2 * Nat.choose y 2 ≤ 200) :=
by sorry

end min_vegetable_dishes_l1216_121692


namespace john_david_pushup_difference_l1216_121652

/-- The number of push-ups done by Zachary -/
def zachary_pushups : ℕ := 19

/-- The number of push-ups David did more than Zachary -/
def david_extra_pushups : ℕ := 39

/-- The number of push-ups done by David -/
def david_pushups : ℕ := 58

/-- The number of push-ups done by John -/
def john_pushups : ℕ := david_pushups

theorem john_david_pushup_difference :
  david_pushups - john_pushups = 0 :=
sorry

end john_david_pushup_difference_l1216_121652


namespace polynomial_inequality_range_l1216_121615

theorem polynomial_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → a * x^3 - x^2 + 4*x + 3 ≥ 0) →
  a ∈ Set.Icc (-6) (-2) := by
sorry

end polynomial_inequality_range_l1216_121615


namespace alex_total_fish_is_4000_l1216_121697

/-- The number of fish Brian catches per trip -/
def brian_fish_per_trip : ℕ := 400

/-- The number of times Chris goes fishing -/
def chris_fishing_trips : ℕ := 10

/-- The number of times Alex goes fishing -/
def alex_fishing_trips : ℕ := chris_fishing_trips / 2

/-- The number of fish Alex catches per trip -/
def alex_fish_per_trip : ℕ := brian_fish_per_trip * 2

/-- The total number of fish Alex caught -/
def alex_total_fish : ℕ := alex_fishing_trips * alex_fish_per_trip

theorem alex_total_fish_is_4000 : alex_total_fish = 4000 := by
  sorry

end alex_total_fish_is_4000_l1216_121697


namespace perfect_squares_among_options_l1216_121674

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem perfect_squares_among_options :
  (is_perfect_square (3^3 * 4^5 * 7^7) = false) ∧
  (is_perfect_square (3^4 * 4^4 * 7^6) = true) ∧
  (is_perfect_square (3^6 * 4^3 * 7^8) = true) ∧
  (is_perfect_square (3^5 * 4^6 * 7^5) = false) ∧
  (is_perfect_square (3^4 * 4^6 * 7^7) = false) :=
by sorry

end perfect_squares_among_options_l1216_121674


namespace total_mulberries_correct_l1216_121694

/-- Represents the mulberry purchase and sale scenario -/
structure MulberrySale where
  total_cost : ℝ
  first_sale_quantity : ℝ
  first_sale_price_increase : ℝ
  second_sale_price_decrease : ℝ
  total_profit : ℝ

/-- Calculates the total amount of mulberries purchased -/
def calculate_total_mulberries (sale : MulberrySale) : ℝ :=
  200 -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the calculated total mulberries is correct -/
theorem total_mulberries_correct (sale : MulberrySale) 
  (h1 : sale.total_cost = 3000)
  (h2 : sale.first_sale_quantity = 150)
  (h3 : sale.first_sale_price_increase = 0.4)
  (h4 : sale.second_sale_price_decrease = 0.2)
  (h5 : sale.total_profit = 750) :
  calculate_total_mulberries sale = 200 := by
  sorry

#eval calculate_total_mulberries {
  total_cost := 3000,
  first_sale_quantity := 150,
  first_sale_price_increase := 0.4,
  second_sale_price_decrease := 0.2,
  total_profit := 750
}

end total_mulberries_correct_l1216_121694


namespace coefficient_sum_l1216_121641

theorem coefficient_sum (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end coefficient_sum_l1216_121641


namespace factor_t_squared_minus_144_l1216_121649

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end factor_t_squared_minus_144_l1216_121649


namespace max_value_constraint_l1216_121676

theorem max_value_constraint (x y z : ℝ) (h : x^2 + 4*y^2 + 9*z^2 = 3) :
  ∃ (M : ℝ), M = 3 ∧ x + 2*y + 3*z ≤ M ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^2 + 4*y₀^2 + 9*z₀^2 = 3 ∧ x₀ + 2*y₀ + 3*z₀ = M :=
by
  sorry

end max_value_constraint_l1216_121676


namespace hyperbola_properties_l1216_121600

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 12 - x^2 / 27 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = (2/3) * x ∨ y = -(2/3) * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y : ℝ, asymptote_equation x y ↔ (y^2 / x^2 = 4/9)) ∧
  hyperbola_equation 3 4 :=
sorry


end hyperbola_properties_l1216_121600


namespace clothing_distribution_l1216_121628

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 47)
  (h2 : first_load = 17)
  (h3 : num_small_loads = 5) :
  (total - first_load) / num_small_loads = 6 := by
  sorry

end clothing_distribution_l1216_121628


namespace square_equation_solutions_l1216_121654

/-- p-arithmetic field -/
structure PArithmetic (p : ℕ) where
  carrier : Type
  zero : carrier
  one : carrier
  add : carrier → carrier → carrier
  mul : carrier → carrier → carrier
  neg : carrier → carrier
  inv : carrier → carrier
  -- Add necessary field axioms here

/-- Definition of squaring in p-arithmetic -/
def square {p : ℕ} (F : PArithmetic p) (x : F.carrier) : F.carrier :=
  F.mul x x

/-- Main theorem: In p-arithmetic (p ≠ 2), x² = a has two distinct solutions for non-zero a -/
theorem square_equation_solutions {p : ℕ} (hp : p ≠ 2) (F : PArithmetic p) :
  ∀ a : F.carrier, a ≠ F.zero →
    ∃ x y : F.carrier, x ≠ y ∧ square F x = a ∧ square F y = a ∧
      ∀ z : F.carrier, square F z = a → (z = x ∨ z = y) :=
sorry

end square_equation_solutions_l1216_121654


namespace g_of_f_minus_two_three_l1216_121684

/-- Transformation f that takes a pair of integers and negates the second component -/
def f (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)

/-- Transformation g that takes a pair of integers and negates both components -/
def g (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, -p.2)

/-- Theorem stating that g[f(-2,3)] = (2,3) -/
theorem g_of_f_minus_two_three : g (f (-2, 3)) = (2, 3) := by
  sorry

end g_of_f_minus_two_three_l1216_121684


namespace biology_homework_wednesday_l1216_121606

def homework_monday : ℚ := 3/5
def remaining_after_monday : ℚ := 1 - homework_monday
def homework_tuesday : ℚ := (1/3) * remaining_after_monday

theorem biology_homework_wednesday :
  1 - homework_monday - homework_tuesday = 4/15 := by
  sorry

end biology_homework_wednesday_l1216_121606


namespace smallest_positive_angle_2012_l1216_121673

/-- Given an angle α = 2012°, this theorem proves that the smallest positive angle 
    with the same terminal side as α is 212°. -/
theorem smallest_positive_angle_2012 (α : Real) (h : α = 2012) :
  ∃ (θ : Real), 0 < θ ∧ θ ≤ 360 ∧ θ = α % 360 ∧ θ = 212 := by
  sorry

end smallest_positive_angle_2012_l1216_121673


namespace largest_integer_satisfying_inequality_l1216_121666

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (1/4 : ℚ) + (x : ℚ)/9 < 1 ↔ x ≤ 6 :=
by sorry

end largest_integer_satisfying_inequality_l1216_121666


namespace coin_flip_problem_l1216_121616

theorem coin_flip_problem (n : ℕ) 
  (p_tails : ℚ) 
  (p_event : ℚ) : 
  p_tails = 1/2 → 
  p_event = 3125/100000 → 
  p_event = (p_tails^2) * ((1 - p_tails)^3) → 
  n ≥ 5 → 
  n = 5 := by
sorry

end coin_flip_problem_l1216_121616


namespace unique_solution_l1216_121611

theorem unique_solution : 
  ∀ (a b : ℕ), 
    a > 1 → 
    b > 0 → 
    b ∣ (a - 1) → 
    (2 * a + 1) ∣ (5 * b - 3) → 
    a = 10 ∧ b = 9 := by
  sorry

end unique_solution_l1216_121611


namespace M_intersect_N_eq_N_l1216_121685

def M : Set Int := {-1, 0, 1}

def N : Set Int := {x | ∃ a b, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end M_intersect_N_eq_N_l1216_121685


namespace teaching_years_difference_l1216_121662

/-- Represents the teaching years of Virginia, Adrienne, and Dennis -/
structure TeachingYears where
  virginia : ℕ
  adrienne : ℕ
  dennis : ℕ

/-- The conditions of the problem -/
def problem_conditions (years : TeachingYears) : Prop :=
  years.virginia + years.adrienne + years.dennis = 75 ∧
  years.virginia = years.adrienne + 9 ∧
  years.dennis = 34

/-- The theorem to be proved -/
theorem teaching_years_difference (years : TeachingYears) 
  (h : problem_conditions years) : 
  years.dennis - years.virginia = 9 := by
  sorry


end teaching_years_difference_l1216_121662


namespace jack_and_jill_speed_l1216_121604

/-- Jack and Jill's Mountain Climb Theorem -/
theorem jack_and_jill_speed (x : ℝ) : 
  (x^2 - 13*x - 26 = (x^2 - 5*x - 66) / (x + 8)) → 
  (x^2 - 13*x - 26 = 4) := by
  sorry

#check jack_and_jill_speed

end jack_and_jill_speed_l1216_121604


namespace car_meeting_problem_l1216_121634

theorem car_meeting_problem (S : ℝ) 
  (h1 : S > 0) 
  (h2 : 60 < S) 
  (h3 : 50 < S) 
  (h4 : (60 / (S - 60)) = ((S - 60 + 50) / (60 + S - 50))) : S = 130 := by
  sorry

end car_meeting_problem_l1216_121634


namespace statement_is_proposition_l1216_121629

-- Define what a proposition is
def is_proposition (statement : String) : Prop :=
  ∃ (truth_value : Bool), (statement = "true") ∨ (statement = "false")

-- Define the statement we want to prove is a proposition
def statement : String := "20-5×3=10"

-- Theorem to prove
theorem statement_is_proposition : is_proposition statement := by
  sorry

end statement_is_proposition_l1216_121629


namespace hexagon_around_convex_curve_l1216_121655

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields/axioms for a convex curve

/-- A hexagon in a 2D plane -/
structure Hexagon where
  -- Add necessary fields for a hexagon (e.g., vertices, sides)

/-- Predicate to check if a hexagon is circumscribed around a convex curve -/
def is_circumscribed (h : Hexagon) (c : ConvexCurve) : Prop :=
  sorry

/-- Predicate to check if all internal angles of a hexagon are equal -/
def has_equal_angles (h : Hexagon) : Prop :=
  sorry

/-- Predicate to check if opposite sides of a hexagon are equal -/
def has_equal_opposite_sides (h : Hexagon) : Prop :=
  sorry

/-- Predicate to check if a hexagon has an axis of symmetry -/
def has_symmetry_axis (h : Hexagon) : Prop :=
  sorry

/-- Theorem: For any convex curve, there exists a circumscribed hexagon with equal angles, 
    equal opposite sides, and an axis of symmetry -/
theorem hexagon_around_convex_curve (c : ConvexCurve) : 
  ∃ h : Hexagon, 
    is_circumscribed h c ∧ 
    has_equal_angles h ∧ 
    has_equal_opposite_sides h ∧ 
    has_symmetry_axis h :=
by
  sorry

end hexagon_around_convex_curve_l1216_121655


namespace sector_area_l1216_121625

/-- Given a circular sector with central angle 2 radians and arc length 4, its area is 4. -/
theorem sector_area (θ : ℝ) (l : ℝ) (r : ℝ) (h1 : θ = 2) (h2 : l = 4) (h3 : l = r * θ) :
  (1 / 2) * r^2 * θ = 4 := by
  sorry

end sector_area_l1216_121625


namespace factorization_equality_minimum_value_minimum_achieved_l1216_121681

-- Problem 1
theorem factorization_equality (m n : ℝ) : 
  m^2 - 4*m*n + 3*n^2 = (m - 3*n) * (m - n) := by sorry

-- Problem 2
theorem minimum_value (m : ℝ) : 
  m^2 - 3*m + 2015 ≥ 2012 + 3/4 := by sorry

-- The minimum is achievable
theorem minimum_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ m : ℝ, m^2 - 3*m + 2015 < 2012 + 3/4 + ε := by sorry

end factorization_equality_minimum_value_minimum_achieved_l1216_121681


namespace apples_picked_total_l1216_121610

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_picked_total :
  total_apples = 11 := by sorry

end apples_picked_total_l1216_121610


namespace shop_e_tv_sets_l1216_121613

theorem shop_e_tv_sets (shops : Fin 5 → ℕ)
  (ha : shops 0 = 20)
  (hb : shops 1 = 30)
  (hc : shops 2 = 60)
  (hd : shops 3 = 80)
  (havg : (shops 0 + shops 1 + shops 2 + shops 3 + shops 4) / 5 = 48) :
  shops 4 = 50 := by
  sorry

end shop_e_tv_sets_l1216_121613


namespace greatest_common_factor_of_three_digit_same_digit_palindromes_l1216_121658

def is_three_digit_same_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ d : ℕ, 1 ≤ d ∧ d ≤ 9 ∧ n = 100 * d + 10 * d + d

theorem greatest_common_factor_of_three_digit_same_digit_palindromes :
  ∃ (gcf : ℕ), gcf = 111 ∧
  (∀ n : ℕ, is_three_digit_same_digit_palindrome n → gcf ∣ n) ∧
  (∀ m : ℕ, (∀ n : ℕ, is_three_digit_same_digit_palindrome n → m ∣ n) → m ≤ gcf) :=
sorry

end greatest_common_factor_of_three_digit_same_digit_palindromes_l1216_121658


namespace no_sequence_with_special_differences_l1216_121644

theorem no_sequence_with_special_differences :
  ¬ ∃ (a : ℕ → ℕ),
    (∀ k : ℕ, ∃! n : ℕ, a (n + 1) - a n = k) ∧
    (∀ k : ℕ, k > 2015 → ∃! n : ℕ, a (n + 2) - a n = k) :=
by sorry

end no_sequence_with_special_differences_l1216_121644


namespace part_one_part_two_l1216_121630

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Part I
theorem part_one : lg 24 - lg 3 - lg 4 + lg 5 = 1 := by sorry

-- Part II
theorem part_two : (((3 : ℝ) ^ (1/3) * (2 : ℝ) ^ (1/2)) ^ 6) + 
                   (((3 : ℝ) * (3 : ℝ) ^ (1/2)) ^ (1/2)) ^ (4/3) - 
                   ((2 : ℝ) ^ (1/4)) * (8 : ℝ) ^ (1/4) - 
                   (2015 : ℝ) ^ 0 = 72 := by sorry

end part_one_part_two_l1216_121630


namespace b_97_mod_81_l1216_121603

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b_97_mod_81 : b 97 ≡ 52 [MOD 81] := by sorry

end b_97_mod_81_l1216_121603


namespace additional_correct_answers_needed_l1216_121653

def total_problems : ℕ := 80
def arithmetic_problems : ℕ := 15
def algebra_problems : ℕ := 25
def geometry_problems : ℕ := 40
def arithmetic_correct_ratio : ℚ := 4/5
def algebra_correct_ratio : ℚ := 1/2
def geometry_correct_ratio : ℚ := 11/20
def passing_grade_ratio : ℚ := 13/20

def correct_answers : ℕ := 
  (arithmetic_problems * arithmetic_correct_ratio).ceil.toNat +
  (algebra_problems * algebra_correct_ratio).ceil.toNat +
  (geometry_problems * geometry_correct_ratio).ceil.toNat

def passing_threshold : ℕ := (total_problems * passing_grade_ratio).ceil.toNat

theorem additional_correct_answers_needed : 
  passing_threshold - correct_answers = 5 := by sorry

end additional_correct_answers_needed_l1216_121653


namespace simplify_fraction_l1216_121677

theorem simplify_fraction (b c : ℚ) (hb : b = 2) (hc : c = 3) :
  15 * b^4 * c^2 / (45 * b^3 * c) = 2 := by sorry

end simplify_fraction_l1216_121677


namespace least_subtraction_for_divisibility_solution_9876543210_and_29_l1216_121679

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  let r := n % d
  (∃ (k : ℕ), (n - r) = d * k) ∧ (∀ (m : ℕ), m < r → ¬(∃ (k : ℕ), (n - m) = d * k)) :=
by
  sorry

theorem solution_9876543210_and_29 :
  let n : ℕ := 9876543210
  let d : ℕ := 29
  let r : ℕ := n % d
  r = 6 ∧
  (∃ (k : ℕ), (n - r) = d * k) ∧
  (∀ (m : ℕ), m < r → ¬(∃ (k : ℕ), (n - m) = d * k)) :=
by
  sorry

end least_subtraction_for_divisibility_solution_9876543210_and_29_l1216_121679


namespace no_subset_with_unique_finite_sum_representation_l1216_121682

-- Define the set S as rational numbers in (0,1)
def S : Set ℚ := {q : ℚ | 0 < q ∧ q < 1}

-- Define the property for subset T
def has_unique_finite_sum_representation (T : Set ℚ) : Prop :=
  ∀ s ∈ S, ∃! (finite_sum : List ℚ),
    (∀ t ∈ finite_sum, t ∈ T) ∧
    (∀ t ∈ finite_sum, ∀ u ∈ finite_sum, t ≠ u → t ≠ u) ∧
    (s = finite_sum.sum)

-- Theorem statement
theorem no_subset_with_unique_finite_sum_representation :
  ¬ ∃ (T : Set ℚ), T ⊆ S ∧ has_unique_finite_sum_representation T := by
  sorry

end no_subset_with_unique_finite_sum_representation_l1216_121682


namespace negative_a_squared_times_a_fourth_l1216_121647

theorem negative_a_squared_times_a_fourth (a : ℝ) : (-a)^2 * a^4 = a^6 := by
  sorry

end negative_a_squared_times_a_fourth_l1216_121647


namespace rest_time_calculation_l1216_121617

theorem rest_time_calculation (walking_rate : ℝ) (total_distance : ℝ) (total_time : ℝ) 
  (rest_interval : ℝ) (h1 : walking_rate = 10) (h2 : total_distance = 50) 
  (h3 : total_time = 320) (h4 : rest_interval = 10) : 
  (total_time - (total_distance / walking_rate) * 60) / ((total_distance / rest_interval) - 1) = 5 := by
  sorry

end rest_time_calculation_l1216_121617


namespace three_numbers_problem_l1216_121639

theorem three_numbers_problem (a b c : ℝ) 
  (sum_eq : a + b + c = 15)
  (sum_minus_third : a + b - c = 10)
  (sum_minus_second : a - b + c = 8) :
  a = 9 ∧ b = 3.5 ∧ c = 2.5 := by
  sorry

end three_numbers_problem_l1216_121639


namespace original_number_proof_l1216_121643

theorem original_number_proof (e : ℝ) : 
  (e * 1.125 - e * 0.75 = 30) → e = 80 := by
  sorry

end original_number_proof_l1216_121643


namespace tree_distance_l1216_121656

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 80) :
  let distance_between (i j : ℕ) := d * (j - i) / 4
  distance_between 1 n = 140 := by
  sorry

end tree_distance_l1216_121656


namespace b_21_equals_861_l1216_121622

def a (n : ℕ) : ℕ := n * (n + 1) / 2

def b (n : ℕ) : ℕ := a (2 * n - 1)

theorem b_21_equals_861 : b 21 = 861 := by sorry

end b_21_equals_861_l1216_121622


namespace expression_simplification_l1216_121693

theorem expression_simplification (x y : ℝ) (hx : x = 2) (hy : y = 2016) :
  (3 * x + 2 * y) * (3 * x - 2 * y) - (x + 2 * y) * (5 * x - 2 * y) / (8 * x) = -2015 :=
by sorry

end expression_simplification_l1216_121693


namespace probability_multiple_of_three_in_eight_rolls_l1216_121619

theorem probability_multiple_of_three_in_eight_rolls : 
  let p : ℚ := 1 - (2/3)^8
  p = 6305/6561 := by sorry

end probability_multiple_of_three_in_eight_rolls_l1216_121619


namespace factorial_divisor_sum_l1216_121608

theorem factorial_divisor_sum (n : ℕ) :
  ∀ k : ℕ, k ≤ n.factorial → ∃ (s : Finset ℕ),
    (∀ x ∈ s, x ∣ n.factorial) ∧
    s.card ≤ n ∧
    k = s.sum id :=
sorry

end factorial_divisor_sum_l1216_121608


namespace smallest_fraction_between_l1216_121651

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < 5 / 8 ∧ 
  (∀ (r s : ℕ+), (3 : ℚ) / 5 < (r : ℚ) / s ∧ (r : ℚ) / s < 5 / 8 → s ≥ q) →
  q - p = 5 := by
  sorry

end smallest_fraction_between_l1216_121651


namespace quadratic_roots_relation_l1216_121660

theorem quadratic_roots_relation (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 5 * x₁ - 7 = 0) → 
  (3 * x₂^2 - 5 * x₂ - 7 = 0) → 
  (x₁ + x₂ = 5/3) ∧ (x₁ * x₂ = -7/3) := by
  sorry

end quadratic_roots_relation_l1216_121660


namespace shakes_undetermined_l1216_121636

/-- Represents the price of a burger -/
def burger_price : ℝ := sorry

/-- Represents the price of a shake -/
def shake_price : ℝ := sorry

/-- Represents the price of a cola -/
def cola_price : ℝ := sorry

/-- Represents the number of shakes in the second purchase -/
def num_shakes_second : ℝ := sorry

/-- The total cost of the first purchase -/
def first_purchase : Prop :=
  3 * burger_price + 7 * shake_price + cola_price = 120

/-- The total cost of the second purchase -/
def second_purchase : Prop :=
  4 * burger_price + num_shakes_second * shake_price + cola_price = 164.5

/-- Theorem stating that the number of shakes in the second purchase cannot be uniquely determined -/
theorem shakes_undetermined (h1 : first_purchase) (h2 : second_purchase) :
  ∃ (x y : ℝ), x ≠ y ∧ 
    (4 * burger_price + x * shake_price + cola_price = 164.5) ∧
    (4 * burger_price + y * shake_price + cola_price = 164.5) :=
  sorry

end shakes_undetermined_l1216_121636


namespace tensor_properties_l1216_121664

/-- Define a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Define the ⊗ operation -/
def tensor (a b : Vector2D) : ℝ :=
  a.x * b.y - b.x * a.y

/-- Define the dot product -/
def dot (a b : Vector2D) : ℝ :=
  a.x * b.x + a.y * b.y

theorem tensor_properties (m n p q : ℝ) :
  let a : Vector2D := ⟨m, n⟩
  let b : Vector2D := ⟨p, q⟩
  (tensor a a = 0) ∧
  ((tensor a b)^2 + (dot a b)^2 = (m^2 + q^2) * (n^2 + p^2)) := by
  sorry

end tensor_properties_l1216_121664


namespace count_elements_with_leftmost_seven_l1216_121691

/-- The set of powers of 5 up to 5000 -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 5000 ∧ n = 5^k}

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The leftmost digit of a natural number -/
def leftmost_digit (n : ℕ) : ℕ := sorry

/-- The count of numbers in S with 7 as the leftmost digit -/
def count_leftmost_seven (S : Set ℕ) : ℕ := sorry

theorem count_elements_with_leftmost_seven :
  num_digits (5^5000) = 3501 →
  leftmost_digit (5^5000) = 7 →
  count_leftmost_seven S = 1501 := by sorry

end count_elements_with_leftmost_seven_l1216_121691


namespace circle_through_point_on_x_axis_l1216_121645

def circle_equation (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

theorem circle_through_point_on_x_axis 
  (center : ℝ × ℝ) 
  (h_center_on_x_axis : center.2 = 0) 
  (h_radius : radius = 1) 
  (h_through_point : (2, 1) ∈ circle_equation center radius) :
  circle_equation center radius = circle_equation (2, 0) 1 := by
sorry

end circle_through_point_on_x_axis_l1216_121645


namespace point_on_line_point_40_161_on_line_l1216_121671

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Given three points on a line, check if a fourth point is on the same line -/
theorem point_on_line (p1 p2 p3 p4 : Point)
  (h1 : collinear p1 p2 p3) : 
  collinear p1 p2 p4 ∧ collinear p2 p3 p4 → collinear p1 p3 p4 := by sorry

/-- The main theorem to prove -/
theorem point_40_161_on_line : 
  let p1 : Point := ⟨2, 9⟩
  let p2 : Point := ⟨6, 25⟩
  let p3 : Point := ⟨10, 41⟩
  let p4 : Point := ⟨40, 161⟩
  collinear p1 p2 p3 → collinear p1 p2 p4 ∧ collinear p2 p3 p4 := by sorry

end point_on_line_point_40_161_on_line_l1216_121671


namespace ratio_a_to_d_l1216_121699

theorem ratio_a_to_d (a b c d : ℚ) : 
  a / b = 8 / 3 →
  b / c = 1 / 5 →
  c / d = 3 / 2 →
  b = 27 →
  a / d = 4 / 5 := by
sorry

end ratio_a_to_d_l1216_121699


namespace arun_weight_average_l1216_121672

def arun_weight_lower_bound : ℝ := 66
def arun_weight_upper_bound : ℝ := 72
def brother_lower_bound : ℝ := 60
def brother_upper_bound : ℝ := 70
def mother_upper_bound : ℝ := 69

theorem arun_weight_average :
  let lower := max arun_weight_lower_bound brother_lower_bound
  let upper := min (min arun_weight_upper_bound brother_upper_bound) mother_upper_bound
  (lower + upper) / 2 = 67.5 := by sorry

end arun_weight_average_l1216_121672


namespace intersection_M_N_l1216_121680

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (x - x^2)}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc 0 1 := by
  sorry

end intersection_M_N_l1216_121680


namespace potato_shipment_l1216_121657

/-- The initial amount of potatoes shipped in kg -/
def initial_potatoes : ℕ := 6500

/-- The amount of damaged potatoes in kg -/
def damaged_potatoes : ℕ := 150

/-- The weight of each bag of potatoes in kg -/
def bag_weight : ℕ := 50

/-- The price of each bag of potatoes in dollars -/
def bag_price : ℕ := 72

/-- The total revenue from selling the potatoes in dollars -/
def total_revenue : ℕ := 9144

theorem potato_shipment :
  initial_potatoes = 
    (total_revenue / bag_price) * bag_weight + damaged_potatoes :=
by sorry

end potato_shipment_l1216_121657


namespace parabola_line_intersection_l1216_121688

/-- The parabola equation: x = 3y^2 + 5y - 4 -/
def parabola (x y : ℝ) : Prop := x = 3 * y^2 + 5 * y - 4

/-- The line equation: x = k -/
def line (x k : ℝ) : Prop := x = k

/-- The condition for a single intersection point -/
def single_intersection (k : ℝ) : Prop :=
  ∃! y, parabola k y ∧ line k k

theorem parabola_line_intersection :
  ∀ k : ℝ, single_intersection k ↔ k = -23/12 := by sorry

end parabola_line_intersection_l1216_121688


namespace ratio_a_to_b_l1216_121635

theorem ratio_a_to_b (a b c d : ℚ) 
  (h1 : b / c = 7 / 9)
  (h2 : c / d = 5 / 7)
  (h3 : a / d = 5 / 12) :
  a / b = 3 / 4 := by
  sorry

end ratio_a_to_b_l1216_121635


namespace noProblemProbabilityIs377Over729_l1216_121640

/-- Recursive function to calculate the number of valid arrangements for n people --/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | (n+2) => 3 * validArrangements (n+1) - validArrangements n

/-- The number of chairs and people --/
def numChairs : ℕ := 6

/-- The total number of possible arrangements --/
def totalArrangements : ℕ := 3^numChairs

/-- The probability of no problematic seating arrangement --/
def noProblemProbability : ℚ := validArrangements numChairs / totalArrangements

theorem noProblemProbabilityIs377Over729 : 
  noProblemProbability = 377 / 729 := by sorry

end noProblemProbabilityIs377Over729_l1216_121640


namespace inequality_solution_l1216_121642

-- Define the inequality function
def f (a : ℝ) (x : ℝ) : Prop := a * x^2 + (1 - a) * x - 1 > 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  if -1 < a ∧ a < 0 then {x | 1 < x ∧ x < -1/a}
  else if a = -1 then ∅
  else if a < -1 then {x | -1/a < x ∧ x < 1}
  else ∅

-- Theorem statement
theorem inequality_solution (a : ℝ) (h : a < 0) :
  {x : ℝ | f a x} = solution_set a :=
sorry

end inequality_solution_l1216_121642


namespace luke_played_two_rounds_l1216_121648

/-- The number of rounds Luke played in a trivia game -/
def rounds_played (total_points : ℕ) (points_per_round : ℕ) : ℕ :=
  total_points / points_per_round

/-- Theorem stating that Luke played 2 rounds -/
theorem luke_played_two_rounds :
  rounds_played 84 42 = 2 := by
  sorry

end luke_played_two_rounds_l1216_121648


namespace intersection_distance_squared_is_675_49_l1216_121669

/-- Two circles in a 2D plane -/
structure TwoCircles where
  center1 : ℝ × ℝ
  radius1 : ℝ
  center2 : ℝ × ℝ
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionDistanceSquared (c : TwoCircles) : ℝ := sorry

/-- The specific configuration of circles from the problem -/
def problemCircles : TwoCircles :=
  { center1 := (3, -1)
  , radius1 := 5
  , center2 := (3, 6)
  , radius2 := 3 }

/-- Theorem stating that the square of the distance between intersection points
    of the given circles is 675/49 -/
theorem intersection_distance_squared_is_675_49 :
  intersectionDistanceSquared problemCircles = 675 / 49 := by sorry

end intersection_distance_squared_is_675_49_l1216_121669


namespace curve_symmetry_condition_l1216_121626

/-- Given a curve y = (mx + n) / (tx + u) symmetric about y = x, prove m - u = 0 -/
theorem curve_symmetry_condition 
  (m n t u : ℝ) 
  (hm : m ≠ 0) (hn : n ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (h_symmetry : ∀ x y : ℝ, y = (m * x + n) / (t * x + u) ↔ x = (m * y + n) / (t * y + u)) :
  m - u = 0 := by
  sorry

end curve_symmetry_condition_l1216_121626


namespace youth_palace_participants_l1216_121614

theorem youth_palace_participants (last_year this_year : ℕ) :
  this_year = last_year + 41 →
  this_year = 3 * last_year - 35 →
  this_year = 79 ∧ last_year = 38 := by
  sorry

end youth_palace_participants_l1216_121614


namespace pascal_triangle_29th_row_28th_number_l1216_121659

theorem pascal_triangle_29th_row_28th_number : Nat.choose 29 27 = 406 := by
  sorry

end pascal_triangle_29th_row_28th_number_l1216_121659


namespace F_and_I_mutually_exclusive_and_complementary_l1216_121601

structure TouristChoice where
  goesToA : Bool
  goesToB : Bool

def E (choice : TouristChoice) : Prop := choice.goesToA ∧ ¬choice.goesToB
def F (choice : TouristChoice) : Prop := choice.goesToA ∨ choice.goesToB
def G (choice : TouristChoice) : Prop := (choice.goesToA ∧ ¬choice.goesToB) ∨ (¬choice.goesToA ∧ choice.goesToB) ∨ (¬choice.goesToA ∧ ¬choice.goesToB)
def H (choice : TouristChoice) : Prop := ¬choice.goesToA
def I (choice : TouristChoice) : Prop := ¬choice.goesToA ∧ ¬choice.goesToB

theorem F_and_I_mutually_exclusive_and_complementary :
  ∀ (choice : TouristChoice),
    (F choice ∧ I choice → False) ∧
    (F choice ∨ I choice) :=
sorry

end F_and_I_mutually_exclusive_and_complementary_l1216_121601


namespace crystal_cupcake_sales_l1216_121620

def crystal_sales (original_cupcake_price original_cookie_price : ℚ)
  (price_reduction_factor : ℚ) (total_revenue : ℚ) (cookies_sold : ℕ) : Prop :=
  let reduced_cupcake_price := original_cupcake_price * price_reduction_factor
  let reduced_cookie_price := original_cookie_price * price_reduction_factor
  let cookie_revenue := reduced_cookie_price * cookies_sold
  let cupcake_revenue := total_revenue - cookie_revenue
  let cupcakes_sold := cupcake_revenue / reduced_cupcake_price
  cupcakes_sold = 16

theorem crystal_cupcake_sales :
  crystal_sales 3 2 (1/2) 32 8 := by sorry

end crystal_cupcake_sales_l1216_121620
