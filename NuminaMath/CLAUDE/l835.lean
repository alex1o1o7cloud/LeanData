import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l835_83591

theorem perpendicular_lines_a_values (a : ℝ) :
  (∀ x y : ℝ, 3 * a * x - y - 1 = 0 ∧ (a - 1) * x + y + 1 = 0 →
    (3 * a * ((a - 1) * x + y + 1) + (-1) * (3 * a * x - y - 1) = 0)) →
  a = -1 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l835_83591


namespace NUMINAMATH_CALUDE_number_of_subjects_l835_83512

theorem number_of_subjects (n : ℕ) 
  (h1 : (75 : ℚ) = (74 * (n - 1) + 80) / n) 
  (h2 : (74 : ℚ) = (75 * n - 80) / (n - 1)) 
  (h3 : n > 1) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_subjects_l835_83512


namespace NUMINAMATH_CALUDE_average_daily_sales_l835_83502

/-- Represents the sales data for a baker's pastry shop over a week. -/
structure BakerSales where
  weekdayPrice : ℕ
  weekendPrice : ℕ
  mondaySales : ℕ
  weekdayIncrease : ℕ
  weekendIncrease : ℕ

/-- Calculates the total pastries sold in a week based on the given sales data. -/
def totalWeeklySales (sales : BakerSales) : ℕ :=
  let tue := sales.mondaySales + sales.weekdayIncrease
  let wed := tue + sales.weekdayIncrease
  let thu := wed + sales.weekdayIncrease
  let fri := thu + sales.weekdayIncrease
  let sat := fri + sales.weekendIncrease
  let sun := sat + sales.weekendIncrease
  sales.mondaySales + tue + wed + thu + fri + sat + sun

/-- Theorem stating that the average daily sales for the given conditions is 59/7. -/
theorem average_daily_sales (sales : BakerSales)
    (h1 : sales.weekdayPrice = 5)
    (h2 : sales.weekendPrice = 6)
    (h3 : sales.mondaySales = 2)
    (h4 : sales.weekdayIncrease = 2)
    (h5 : sales.weekendIncrease = 3) :
    (totalWeeklySales sales : ℚ) / 7 = 59 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_sales_l835_83502


namespace NUMINAMATH_CALUDE_l_shaped_area_is_ten_l835_83597

/-- The area of an L-shaped region formed by subtracting four smaller squares from a larger square -/
def l_shaped_area (large_side : ℝ) (small_side1 small_side2 : ℝ) : ℝ :=
  large_side^2 - 2 * small_side1^2 - 2 * small_side2^2

/-- Theorem stating that the area of the L-shaped region is 10 -/
theorem l_shaped_area_is_ten :
  l_shaped_area 6 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_is_ten_l835_83597


namespace NUMINAMATH_CALUDE_line_equation_proof_l835_83539

-- Define the line l
def line_l : Set (ℝ × ℝ) := {(x, y) | 4*x - 3*y - 1 = 0}

-- Define the given line
def given_line : Set (ℝ × ℝ) := {(x, y) | 3*x + 4*y - 3 = 0}

-- Define the point A
def point_A : ℝ × ℝ := (-2, -3)

theorem line_equation_proof :
  -- Line l passes through point A
  point_A ∈ line_l ∧
  -- Line l is perpendicular to the given line
  (∀ (p q : ℝ × ℝ), p ∈ line_l → q ∈ line_l → p ≠ q →
    ∀ (r s : ℝ × ℝ), r ∈ given_line → s ∈ given_line → r ≠ s →
      ((p.1 - q.1) * (r.1 - s.1) + (p.2 - q.2) * (r.2 - s.2) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l835_83539


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l835_83583

/-- Definition of the bow-tie operation -/
noncomputable def bowtie (p q : ℝ) : ℝ := p + Real.sqrt (q + Real.sqrt (q + Real.sqrt (q + Real.sqrt q)))

/-- Theorem: If 5 bow-tie q equals 13, then q equals 56 -/
theorem bowtie_equation_solution (q : ℝ) : bowtie 5 q = 13 → q = 56 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l835_83583


namespace NUMINAMATH_CALUDE_candy_pencils_count_l835_83554

/-- The number of pencils Candy has -/
def candy_pencils : ℕ := 9

/-- The number of pencils Caleb has -/
def caleb_pencils : ℕ := 2 * candy_pencils - 3

/-- The original number of pencils Calen had -/
def calen_original_pencils : ℕ := caleb_pencils + 5

/-- The number of pencils Calen lost -/
def calen_lost_pencils : ℕ := 10

/-- The number of pencils Calen has now -/
def calen_current_pencils : ℕ := 10

theorem candy_pencils_count :
  calen_original_pencils - calen_lost_pencils = calen_current_pencils :=
by sorry

end NUMINAMATH_CALUDE_candy_pencils_count_l835_83554


namespace NUMINAMATH_CALUDE_cubic_function_property_l835_83585

theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x - 4
  f (-2) = 2 → f 2 = -10 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l835_83585


namespace NUMINAMATH_CALUDE_fewer_sevens_to_hundred_l835_83517

theorem fewer_sevens_to_hundred : ∃ (n : ℕ) (expr : ℕ → ℚ), 
  n < 10 ∧ 
  (∀ i, i < n → expr i = 7) ∧
  (∃ f : (Fin n → ℚ) → ℚ, f (λ i => expr i) = 100) :=
sorry

end NUMINAMATH_CALUDE_fewer_sevens_to_hundred_l835_83517


namespace NUMINAMATH_CALUDE_solve_system_for_y_l835_83513

theorem solve_system_for_y (x y : ℚ) 
  (eq1 : 2 * x - 3 * y = 5)
  (eq2 : 4 * x + 9 * y = 6) :
  y = -4/15 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_y_l835_83513


namespace NUMINAMATH_CALUDE_nicki_total_miles_run_l835_83584

/-- Calculates the total miles run in a year given weekly mileage for each half -/
def total_miles_run (weeks_in_year : ℕ) (miles_first_half : ℕ) (miles_second_half : ℕ) : ℕ :=
  let half_year := weeks_in_year / 2
  (miles_first_half * half_year) + (miles_second_half * half_year)

theorem nicki_total_miles_run : total_miles_run 52 20 30 = 1300 := by
  sorry

#eval total_miles_run 52 20 30

end NUMINAMATH_CALUDE_nicki_total_miles_run_l835_83584


namespace NUMINAMATH_CALUDE_dans_song_book_cost_l835_83533

/-- The cost of Dan's song book is equal to the total amount spent at the music store
    minus the cost of the clarinet. -/
theorem dans_song_book_cost (clarinet_cost total_spent : ℚ) 
  (h1 : clarinet_cost = 130.30)
  (h2 : total_spent = 141.54) :
  total_spent - clarinet_cost = 11.24 := by
  sorry

end NUMINAMATH_CALUDE_dans_song_book_cost_l835_83533


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_l835_83542

/-- The cost of paint per kg for a cube with given conditions -/
theorem paint_cost_per_kg (coverage : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  coverage = 16 →
  total_cost = 876 →
  side_length = 8 →
  (total_cost / (6 * side_length^2 / coverage)) = 36.5 :=
by sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_l835_83542


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_x_l835_83540

/-- A polynomial with non-negative integer coefficients -/
def NonNegIntPoly := ℕ → ℕ

/-- Definition of x_n -/
def x (P Q : NonNegIntPoly) (n : ℕ) : ℕ := 2016^(P n) + Q n

/-- A number is squarefree if it's not divisible by any prime square -/
def IsSquarefree (m : ℕ) : Prop := ∀ p : ℕ, Nat.Prime p → (p^2 ∣ m) → False

theorem infinite_primes_dividing_x (P Q : NonNegIntPoly) 
  (hP : ¬ ∀ n : ℕ, P n = P 0) 
  (hQ : ¬ ∀ n : ℕ, Q n = Q 0) : 
  ∃ S : Set ℕ, (S.Infinite) ∧ 
  (∀ p ∈ S, Nat.Prime p ∧ ∃ m : ℕ, IsSquarefree m ∧ (p ∣ x P Q m)) := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_x_l835_83540


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l835_83599

theorem binomial_coefficient_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃ + a₅) = -(122 / 121) := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l835_83599


namespace NUMINAMATH_CALUDE_socks_in_washing_machine_l835_83559

/-- The number of players in a soccer match -/
def num_players : ℕ := 11

/-- The number of socks each player wears -/
def socks_per_player : ℕ := 2

/-- The total number of socks in the washing machine -/
def total_socks : ℕ := num_players * socks_per_player

theorem socks_in_washing_machine : total_socks = 22 := by
  sorry

end NUMINAMATH_CALUDE_socks_in_washing_machine_l835_83559


namespace NUMINAMATH_CALUDE_inequality_proof_l835_83595

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a < c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l835_83595


namespace NUMINAMATH_CALUDE_slinkums_shipment_correct_l835_83575

/-- The total number of Mr. Slinkums in the initial shipment -/
def total_slinkums : ℕ := 200

/-- The percentage of Mr. Slinkums on display -/
def display_percentage : ℚ := 30 / 100

/-- The number of Mr. Slinkums in storage -/
def storage_slinkums : ℕ := 140

/-- Theorem stating that the total number of Mr. Slinkums is correct given the conditions -/
theorem slinkums_shipment_correct : 
  (1 - display_percentage) * (total_slinkums : ℚ) = storage_slinkums := by
  sorry

end NUMINAMATH_CALUDE_slinkums_shipment_correct_l835_83575


namespace NUMINAMATH_CALUDE_fish_tank_problem_l835_83568

theorem fish_tank_problem (x : ℕ) : x + (x - 4) = 20 → x - 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l835_83568


namespace NUMINAMATH_CALUDE_simple_interest_rate_l835_83598

theorem simple_interest_rate : 
  ∀ (P : ℝ) (R : ℝ),
  P > 0 →
  (P * R * 10) / 100 = (3 / 5) * P →
  R = 6 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l835_83598


namespace NUMINAMATH_CALUDE_three_billion_three_hundred_million_scientific_notation_l835_83569

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem three_billion_three_hundred_million_scientific_notation :
  to_scientific_notation 3300000000 = ScientificNotation.mk 3.3 9 sorry := by
  sorry

end NUMINAMATH_CALUDE_three_billion_three_hundred_million_scientific_notation_l835_83569


namespace NUMINAMATH_CALUDE_rotation_of_D_l835_83581

/-- Rotates a point 90 degrees clockwise about the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotation_of_D : 
  let D : ℝ × ℝ := (-3, -8)
  rotate90Clockwise D = (-8, 3) := by
sorry

end NUMINAMATH_CALUDE_rotation_of_D_l835_83581


namespace NUMINAMATH_CALUDE_solution_set_inequality_l835_83580

theorem solution_set_inequality (x : ℝ) : 
  (2 * x) / (x - 1) < 1 ↔ -1 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l835_83580


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_three_halves_l835_83531

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined as (1+2i)(3+ai) where a is a real number. -/
def z (a : ℝ) : ℂ := (1 + 2*Complex.I) * (3 + a*Complex.I)

/-- If z is purely imaginary, then a = 3/2. -/
theorem purely_imaginary_implies_a_eq_three_halves :
  ∀ a : ℝ, IsPurelyImaginary (z a) → a = 3/2 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_three_halves_l835_83531


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l835_83535

/-- A polynomial of degree 103 with real coefficients -/
def poly (C D : ℝ) (x : ℂ) : ℂ := x^103 + C*x + D

/-- The quadratic polynomial x^2 - x + 1 -/
def quad (x : ℂ) : ℂ := x^2 - x + 1

theorem polynomial_division_theorem (C D : ℝ) :
  (∀ x : ℂ, quad x = 0 → poly C D x = 0) →
  C + D = -1 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l835_83535


namespace NUMINAMATH_CALUDE_cookie_theorem_l835_83509

def cookie_problem (initial_cookies : ℕ) (given_to_friend : ℕ) (eaten : ℕ) : ℕ :=
  let remaining_after_friend := initial_cookies - given_to_friend
  let given_to_family := remaining_after_friend / 2
  let remaining_after_family := remaining_after_friend - given_to_family
  remaining_after_family - eaten

theorem cookie_theorem : cookie_problem 19 5 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_theorem_l835_83509


namespace NUMINAMATH_CALUDE_carwash_solution_l835_83530

/-- Represents the carwash problem --/
structure CarWash where
  car_price : ℕ
  truck_price : ℕ
  suv_price : ℕ
  total_raised : ℕ
  num_suvs : ℕ
  num_trucks : ℕ

/-- Calculates the number of cars washed --/
def cars_washed (cw : CarWash) : ℕ :=
  (cw.total_raised - (cw.suv_price * cw.num_suvs + cw.truck_price * cw.num_trucks)) / cw.car_price

/-- Theorem stating the solution to the carwash problem --/
theorem carwash_solution (cw : CarWash) 
  (h1 : cw.car_price = 5)
  (h2 : cw.truck_price = 6)
  (h3 : cw.suv_price = 7)
  (h4 : cw.total_raised = 100)
  (h5 : cw.num_suvs = 5)
  (h6 : cw.num_trucks = 5) :
  cars_washed cw = 7 := by
  sorry

#eval cars_washed ⟨5, 6, 7, 100, 5, 5⟩

end NUMINAMATH_CALUDE_carwash_solution_l835_83530


namespace NUMINAMATH_CALUDE_bryce_raisins_l835_83527

theorem bryce_raisins : ∃ (b c : ℕ), b = c + 8 ∧ c = b / 3 → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_bryce_raisins_l835_83527


namespace NUMINAMATH_CALUDE_initial_ribbon_tape_length_l835_83545

/-- The initial length of ribbon tape Yujin had, in meters. -/
def initial_length : ℝ := 8.9

/-- The length of ribbon tape required for one ribbon, in meters. -/
def ribbon_length : ℝ := 0.84

/-- The number of ribbons made. -/
def num_ribbons : ℕ := 10

/-- The length of remaining ribbon tape, in meters. -/
def remaining_length : ℝ := 0.5

/-- Theorem stating that the initial length of ribbon tape equals 8.9 meters. -/
theorem initial_ribbon_tape_length :
  initial_length = ribbon_length * num_ribbons + remaining_length := by
  sorry

end NUMINAMATH_CALUDE_initial_ribbon_tape_length_l835_83545


namespace NUMINAMATH_CALUDE_sec_330_deg_l835_83503

/-- Prove that sec 330° = 2√3 / 3 -/
theorem sec_330_deg : 
  let sec : Real → Real := λ θ ↦ 1 / Real.cos θ
  let θ : Real := 330 * Real.pi / 180
  sec θ = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sec_330_deg_l835_83503


namespace NUMINAMATH_CALUDE_no_consecutive_sum_32_l835_83556

theorem no_consecutive_sum_32 : ¬∃ (n k : ℕ), n > 0 ∧ (n * (2 * k + n - 1)) / 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_sum_32_l835_83556


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l835_83570

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem fifteenth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 17) (h₃ : a₃ = 31) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 199 := by
  sorry

#check fifteenth_term_of_sequence

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l835_83570


namespace NUMINAMATH_CALUDE_forgotten_poems_sally_forgotten_poems_l835_83563

/-- Given the number of initially memorized poems and the number of poems that can be recited,
    prove that the number of forgotten poems is their difference. -/
theorem forgotten_poems (initially_memorized recitable : ℕ) :
  initially_memorized ≥ recitable →
  initially_memorized - recitable = initially_memorized - recitable :=
by
  sorry

/-- Application to Sally's specific case -/
theorem sally_forgotten_poems :
  let initially_memorized := 8
  let recitable := 3
  initially_memorized - recitable = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_forgotten_poems_sally_forgotten_poems_l835_83563


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l835_83505

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l835_83505


namespace NUMINAMATH_CALUDE_complete_square_formula_not_complete_square_A_not_complete_square_B_not_complete_square_C_l835_83553

theorem complete_square_formula (a b : ℝ) : 
  (a - b) * (b - a) = -(a - b)^2 :=
sorry

theorem not_complete_square_A (a b : ℝ) :
  (a - b) * (a + b) = a^2 - b^2 :=
sorry

theorem not_complete_square_B (a b : ℝ) :
  -(a + b) * (b - a) = a^2 - b^2 :=
sorry

theorem not_complete_square_C (a b : ℝ) :
  (a + b) * (b - a) = b^2 - a^2 :=
sorry

end NUMINAMATH_CALUDE_complete_square_formula_not_complete_square_A_not_complete_square_B_not_complete_square_C_l835_83553


namespace NUMINAMATH_CALUDE_two_roots_condition_l835_83582

-- Define the equation
def f (x a : ℝ) : ℝ := 4 * x^2 - 16 * |x| + (2 * a + |x| - x)^2 - 16

-- Define the condition for exactly two distinct roots
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0 ∧
  ∀ x : ℝ, f x a = 0 → x = x₁ ∨ x = x₂

-- State the theorem
theorem two_roots_condition :
  ∀ a : ℝ, has_two_distinct_roots a ↔ (a > -6 ∧ a ≤ -2) ∨ (a > 2 ∧ a < Real.sqrt 8) :=
sorry

end NUMINAMATH_CALUDE_two_roots_condition_l835_83582


namespace NUMINAMATH_CALUDE_gcd_210_378_l835_83526

theorem gcd_210_378 : Nat.gcd 210 378 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcd_210_378_l835_83526


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l835_83523

/-- Profit percentage calculation for Company N --/
theorem profit_percentage_calculation (R : ℝ) (P : ℝ) :
  R > 0 ∧ P > 0 →  -- Assuming positive revenue and profit
  (0.8 * R) * 0.14 = 0.112 * R →  -- 1999 profit calculation
  0.112 * R = 1.1200000000000001 * P →  -- Profit comparison between years
  P / R * 100 = 10 := by
sorry


end NUMINAMATH_CALUDE_profit_percentage_calculation_l835_83523


namespace NUMINAMATH_CALUDE_triangle_ratio_l835_83534

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  2 * (Real.cos (A / 2))^2 = (Real.sqrt 3 / 3) * Real.sin A →
  Real.sin (B - C) = 4 * Real.cos B * Real.sin C →
  b / c = 1 + Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l835_83534


namespace NUMINAMATH_CALUDE_at_least_100_odd_population_days_l835_83525

/-- Represents the state of the Martian population on a given day -/
structure PopulationState :=
  (day : ℕ)
  (births : ℕ)
  (population : ℕ)

/-- A function that calculates the population state for each day -/
def populationEvolution : ℕ → PopulationState → PopulationState :=
  sorry

/-- The total number of Martians born throughout history -/
def totalBirths : ℕ := sorry

/-- Theorem stating that there are at least 100 days with odd population -/
theorem at_least_100_odd_population_days
  (h_odd_births : Odd totalBirths)
  (h_lifespan : ∀ (m : ℕ), m < totalBirths → ∃ (b d : ℕ), d - b = 100 ∧ PopulationState.population (populationEvolution d (PopulationState.mk b 1 1)) = PopulationState.population (populationEvolution (d + 1) (PopulationState.mk b 1 1)) - 1) :
  ∃ (S : Finset ℕ), S.card ≥ 100 ∧ ∀ (d : ℕ), d ∈ S → Odd (PopulationState.population (populationEvolution d (PopulationState.mk 0 0 0))) :=
sorry

end NUMINAMATH_CALUDE_at_least_100_odd_population_days_l835_83525


namespace NUMINAMATH_CALUDE_congruent_triangles_on_skew_lines_l835_83590

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew (g l : Line3D) : Prop := sorry

/-- A point lies on a line in 3D space. -/
def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

/-- Two triangles in 3D space are congruent. -/
def triangles_congruent (t1 t2 : Triangle3D) : Prop := sorry

/-- The number of congruent triangles that can be constructed on two skew lines. -/
def num_congruent_triangles_on_skew_lines (g l : Line3D) (abc : Triangle3D) : ℕ := sorry

/-- Theorem: Given two skew lines and a triangle, there exist exactly 16 congruent triangles
    with vertices on the given lines. -/
theorem congruent_triangles_on_skew_lines (g l : Line3D) (abc : Triangle3D) :
  are_skew g l →
  num_congruent_triangles_on_skew_lines g l abc = 16 :=
by sorry

end NUMINAMATH_CALUDE_congruent_triangles_on_skew_lines_l835_83590


namespace NUMINAMATH_CALUDE_amy_initial_amount_l835_83567

/-- The amount of money Amy had when she got to the fair -/
def initial_amount : ℕ := sorry

/-- The amount of money Amy had when she left the fair -/
def final_amount : ℕ := 11

/-- The amount of money Amy spent at the fair -/
def spent_amount : ℕ := 4

/-- Theorem: Amy had $15 when she got to the fair -/
theorem amy_initial_amount : initial_amount = 15 := by
  sorry

end NUMINAMATH_CALUDE_amy_initial_amount_l835_83567


namespace NUMINAMATH_CALUDE_simplify_expression_l835_83504

theorem simplify_expression (x : ℝ) : (2*x)^5 + (4*x)*(x^4) + 5*x^3 = 36*x^5 + 5*x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l835_83504


namespace NUMINAMATH_CALUDE_comet_watch_percentage_l835_83532

-- Define the total time spent on activities in minutes
def total_time : ℕ := 655

-- Define the time spent watching the comet in minutes
def comet_watch_time : ℕ := 20

-- Function to calculate percentage
def calculate_percentage (part : ℕ) (whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- Function to round to nearest integer
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

-- Theorem statement
theorem comet_watch_percentage :
  round_to_nearest (calculate_percentage comet_watch_time total_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_comet_watch_percentage_l835_83532


namespace NUMINAMATH_CALUDE_train_length_l835_83571

/-- The length of a train given its relative speed and passing time -/
theorem train_length (relative_speed : ℝ) (passing_time : ℝ) : 
  relative_speed = 72 - 36 →
  passing_time = 12 →
  relative_speed * (1000 / 3600) * passing_time = 120 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l835_83571


namespace NUMINAMATH_CALUDE_vector_magnitude_l835_83516

theorem vector_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 1)
  (h2 : ‖b‖ = 2)
  (h3 : ‖a - b‖ = Real.sqrt 3) :
  ‖a + b‖ = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l835_83516


namespace NUMINAMATH_CALUDE_expression_equality_l835_83562

theorem expression_equality : 
  |Real.sqrt 8 - 2| + (π - 2023)^(0 : ℝ) + (-1/2)^(-2 : ℝ) - 2 * Real.cos (60 * π / 180) = 2 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l835_83562


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l835_83576

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l835_83576


namespace NUMINAMATH_CALUDE_division_problem_l835_83561

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 690 →
  divisor = 36 →
  remainder = 6 →
  dividend = divisor * quotient + remainder →
  quotient = 19 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l835_83561


namespace NUMINAMATH_CALUDE_two_decimals_sum_and_difference_l835_83574

theorem two_decimals_sum_and_difference (x y : ℝ) : 
  (0 < x ∧ x < 10 ∧ 0 < y ∧ y < 10) → -- x and y are single-digit decimals
  (x + y = 10) →                     -- their sum is 10
  (|x - y| = 0.4) →                  -- their difference is 0.4
  ((x = 4.8 ∧ y = 5.2) ∨ (x = 5.2 ∧ y = 4.8)) := by
sorry

end NUMINAMATH_CALUDE_two_decimals_sum_and_difference_l835_83574


namespace NUMINAMATH_CALUDE_fifth_number_eighth_row_l835_83538

/-- Represents the end number of the n-th row in the table -/
def end_of_row (n : ℕ) : ℕ := n * n

/-- Represents the first number in the n-th row -/
def start_of_row (n : ℕ) : ℕ := end_of_row (n - 1) + 1

/-- The theorem stating that the 5th number from the left in the 8th row is 54 -/
theorem fifth_number_eighth_row : start_of_row 8 + 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_eighth_row_l835_83538


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l835_83566

/-- A linear function that passes through the first, third, and fourth quadrants -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k + 1) * x + k - 2

/-- Condition for the function to have a positive slope -/
def positive_slope (k : ℝ) : Prop := k + 1 > 0

/-- Condition for the y-intercept to be negative -/
def negative_y_intercept (k : ℝ) : Prop := k - 2 < 0

/-- Theorem stating the range of k for the linear function to pass through the first, third, and fourth quadrants -/
theorem linear_function_quadrants (k : ℝ) : 
  (∀ x, ∃ y, y = linear_function k x) ∧ 
  positive_slope k ∧ 
  negative_y_intercept k ↔ 
  -1 < k ∧ k < 2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l835_83566


namespace NUMINAMATH_CALUDE_smallest_fraction_above_four_fifths_l835_83519

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_fraction_above_four_fifths :
  ∀ (a b : ℕ), is_two_digit a → is_two_digit b → (a : ℚ) / b > 4 / 5 → Nat.gcd a b = 1 →
  (77 : ℚ) / 96 ≤ (a : ℚ) / b :=
sorry

end NUMINAMATH_CALUDE_smallest_fraction_above_four_fifths_l835_83519


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l835_83565

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l835_83565


namespace NUMINAMATH_CALUDE_specific_coin_expected_value_l835_83588

/-- A biased coin with probabilities for heads and tails, and associated winnings/losses. -/
structure BiasedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  win_heads : ℚ
  loss_tails : ℚ

/-- Expected value of winnings for a single flip of a biased coin. -/
def expected_value (coin : BiasedCoin) : ℚ :=
  coin.prob_heads * coin.win_heads + coin.prob_tails * (-coin.loss_tails)

/-- Theorem stating the expected value for the specific coin in the problem. -/
theorem specific_coin_expected_value :
  let coin : BiasedCoin := {
    prob_heads := 1/4,
    prob_tails := 3/4,
    win_heads := 4,
    loss_tails := 3
  }
  expected_value coin = -5/4 := by sorry

end NUMINAMATH_CALUDE_specific_coin_expected_value_l835_83588


namespace NUMINAMATH_CALUDE_distinct_cube_digits_mod_seven_l835_83589

theorem distinct_cube_digits_mod_seven :
  ∃! s : Finset ℕ, 
    (∀ n : ℕ, (n^3 % 7) ∈ s) ∧ 
    (∀ m ∈ s, ∃ n : ℕ, n^3 % 7 = m) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_cube_digits_mod_seven_l835_83589


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l835_83573

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (3 - Complex.I) / (2 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l835_83573


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l835_83564

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt ((3 / x) + 5) = 5/2 → x = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l835_83564


namespace NUMINAMATH_CALUDE_watermelons_last_two_weeks_l835_83577

/-- Represents the number of watermelons Jeremy eats in a given week -/
def jeremyEats (week : ℕ) : ℕ :=
  match week % 3 with
  | 0 => 3
  | 1 => 4
  | _ => 5

/-- Represents the number of watermelons Jeremy gives to his dad in a given week -/
def dadReceives (week : ℕ) : ℕ := week + 1

/-- Represents the number of watermelons Jeremy gives to his sister in a given week -/
def sisterReceives (week : ℕ) : ℕ := 2 * week - 1

/-- Represents the number of watermelons Jeremy gives to his neighbor in a given week -/
def neighborReceives (week : ℕ) : ℕ := max (2 - week) 0

/-- Represents the total number of watermelons consumed in a given week -/
def totalConsumed (week : ℕ) : ℕ :=
  jeremyEats week + dadReceives week + sisterReceives week + neighborReceives week

/-- The initial number of watermelons -/
def initialWatermelons : ℕ := 30

/-- Theorem stating that the watermelons will last for 2 complete weeks -/
theorem watermelons_last_two_weeks :
  initialWatermelons ≥ totalConsumed 1 + totalConsumed 2 ∧
  initialWatermelons < totalConsumed 1 + totalConsumed 2 + totalConsumed 3 :=
sorry

end NUMINAMATH_CALUDE_watermelons_last_two_weeks_l835_83577


namespace NUMINAMATH_CALUDE_closest_to_product_l835_83544

def product : ℝ := 0.001532 * 2134672

def options : List ℝ := [3100, 3150, 3200, 3500, 4000]

theorem closest_to_product : 
  ∃ (x : ℝ), x ∈ options ∧ 
  ∀ (y : ℝ), y ∈ options → |product - x| ≤ |product - y| ∧
  x = 3150 :=
sorry

end NUMINAMATH_CALUDE_closest_to_product_l835_83544


namespace NUMINAMATH_CALUDE_sqrt_5_simplest_l835_83560

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x = Real.sqrt y → ¬∃ (a b : ℚ), y = a / b ∧ b ≠ 1

theorem sqrt_5_simplest :
  is_simplest_sqrt (Real.sqrt 5) ∧
  ¬is_simplest_sqrt (Real.sqrt 2.5) ∧
  ¬is_simplest_sqrt (Real.sqrt 8) ∧
  ¬is_simplest_sqrt (Real.sqrt (1/3)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_5_simplest_l835_83560


namespace NUMINAMATH_CALUDE_dice_product_120_probability_l835_83552

/-- A function representing a standard die roll --/
def standardDie : ℕ → Prop :=
  λ n => 1 ≤ n ∧ n ≤ 6

/-- The probability of a specific outcome when rolling three dice --/
def tripleRollProb : ℚ := (1 : ℚ) / 216

/-- The number of favorable outcomes --/
def favorableOutcomes : ℕ := 6

/-- The probability that the product of three dice rolls equals 120 --/
theorem dice_product_120_probability :
  (favorableOutcomes : ℚ) * tripleRollProb = (1 : ℚ) / 36 :=
sorry

end NUMINAMATH_CALUDE_dice_product_120_probability_l835_83552


namespace NUMINAMATH_CALUDE_golf_carts_needed_l835_83549

theorem golf_carts_needed (patrons_per_cart : ℕ) (car_patrons : ℕ) (bus_patrons : ℕ) : 
  patrons_per_cart = 3 →
  car_patrons = 12 →
  bus_patrons = 27 →
  ((car_patrons + bus_patrons) + patrons_per_cart - 1) / patrons_per_cart = 13 := by
sorry

end NUMINAMATH_CALUDE_golf_carts_needed_l835_83549


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l835_83551

theorem cricket_team_average_age (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) :
  team_size = 11 →
  captain_age = 26 →
  wicket_keeper_age_diff = 3 →
  let total_age := team_size * (captain_age + wicket_keeper_age_diff + 2) / 2
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + captain_age + wicket_keeper_age_diff)
  (remaining_age / remaining_players) + 1 = total_age / team_size →
  total_age / team_size = 32 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l835_83551


namespace NUMINAMATH_CALUDE_bank_queue_properties_l835_83514

/-- Represents a bank queue with simple and long operations -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected wasted person-minutes assuming random order -/
def expected_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Theorem stating the properties of the bank queue problem -/
theorem bank_queue_properties (q : BankQueue) 
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧ 
  max_wasted_time q = 100 ∧
  expected_wasted_time q = 84 :=
by sorry

end NUMINAMATH_CALUDE_bank_queue_properties_l835_83514


namespace NUMINAMATH_CALUDE_soda_preference_result_l835_83557

/-- The number of people who prefer calling soft drinks "Soda" in a survey. -/
def soda_preference (total_surveyed : ℕ) (central_angle : ℕ) : ℕ :=
  (total_surveyed * central_angle) / 360

/-- Theorem stating that 330 people prefer calling soft drinks "Soda" in the given survey. -/
theorem soda_preference_result : soda_preference 600 198 = 330 := by
  sorry

end NUMINAMATH_CALUDE_soda_preference_result_l835_83557


namespace NUMINAMATH_CALUDE_rectangular_field_area_l835_83528

theorem rectangular_field_area (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width > 0 → 
  length > 0 → 
  width = length / 3 → 
  perimeter = 2 * (width + length) → 
  perimeter = 72 → 
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l835_83528


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l835_83536

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 5 = 12)
  (h_second : a 2 = 3) :
  a 6 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l835_83536


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l835_83579

theorem complex_fraction_simplification :
  ((3 + 2*Complex.I) / (2 - 3*Complex.I)) - ((3 - 2*Complex.I) / (2 + 3*Complex.I)) = 2*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l835_83579


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l835_83550

theorem simplify_and_evaluate (x : ℝ) (h : x = 2) :
  (1 + 1/x) / ((x^2 - 1) / x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l835_83550


namespace NUMINAMATH_CALUDE_danai_decorations_l835_83506

/-- The number of decorations Danai will put up in total -/
def total_decorations (skulls broomsticks spiderwebs cauldrons additional_budget additional_left : ℕ) : ℕ :=
  skulls + broomsticks + spiderwebs + 2 * spiderwebs + cauldrons + additional_budget + additional_left

/-- Theorem stating the total number of decorations Danai will put up -/
theorem danai_decorations :
  total_decorations 12 4 12 1 20 10 = 83 := by
  sorry

end NUMINAMATH_CALUDE_danai_decorations_l835_83506


namespace NUMINAMATH_CALUDE_min_value_theorem_l835_83529

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (1 / x + 2 / y) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l835_83529


namespace NUMINAMATH_CALUDE_nesbitt_inequality_l835_83586

theorem nesbitt_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_nesbitt_inequality_l835_83586


namespace NUMINAMATH_CALUDE_abs_neg_reciprocal_of_two_l835_83500

theorem abs_neg_reciprocal_of_two : |-(1 / 2)| = |1 / 2| := by sorry

end NUMINAMATH_CALUDE_abs_neg_reciprocal_of_two_l835_83500


namespace NUMINAMATH_CALUDE_rescue_team_distribution_l835_83592

/-- The number of ways to distribute rescue teams to disaster sites. -/
def distribute_teams (total_teams : ℕ) (num_sites : ℕ) : ℕ :=
  sorry

/-- Constraint that each site gets at least one team -/
def at_least_one_each (distribution : List ℕ) : Prop :=
  sorry

/-- Constraint that site A gets at least two teams -/
def site_A_at_least_two (distribution : List ℕ) : Prop :=
  sorry

theorem rescue_team_distribution :
  ∃ (distributions : List (List ℕ)),
    (∀ d ∈ distributions,
      d.length = 3 ∧
      d.sum = 6 ∧
      at_least_one_each d ∧
      site_A_at_least_two d) ∧
    distributions.length = 360 :=
  sorry

end NUMINAMATH_CALUDE_rescue_team_distribution_l835_83592


namespace NUMINAMATH_CALUDE_new_variance_after_adding_datapoint_l835_83511

/-- Given a sample with size 7, average 5, and variance 2, adding a new data point of 5 results in a new variance of 7/4 -/
theorem new_variance_after_adding_datapoint
  (sample_size : ℕ)
  (original_avg : ℝ)
  (original_var : ℝ)
  (new_datapoint : ℝ)
  (h1 : sample_size = 7)
  (h2 : original_avg = 5)
  (h3 : original_var = 2)
  (h4 : new_datapoint = 5) :
  let new_sample_size : ℕ := sample_size + 1
  let new_avg : ℝ := (sample_size * original_avg + new_datapoint) / new_sample_size
  let new_var : ℝ := (sample_size * original_var + sample_size * (new_avg - original_avg)^2) / new_sample_size
  new_var = 7/4 := by sorry

end NUMINAMATH_CALUDE_new_variance_after_adding_datapoint_l835_83511


namespace NUMINAMATH_CALUDE_zero_point_implies_a_range_l835_83594

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- State the theorem
theorem zero_point_implies_a_range (a : ℝ) :
  (∃ x : ℝ, 1/2 < x ∧ x < 4 ∧ f a x = 0) → 2 ≤ a ∧ a < 17/4 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_implies_a_range_l835_83594


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l835_83547

/-- The nth term of a geometric sequence -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 10th term of a geometric sequence with first term 4 and common ratio 5/3 -/
theorem tenth_term_of_specific_sequence :
  geometric_sequence 4 (5/3) 10 = 7812500/19683 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l835_83547


namespace NUMINAMATH_CALUDE_equation_solutions_l835_83558

theorem equation_solutions : 
  {x : ℝ | x^4 + (3-x)^4 + x^3 = 82} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l835_83558


namespace NUMINAMATH_CALUDE_smallest_fifth_prime_term_l835_83515

/-- An arithmetic sequence of five prime numbers -/
structure PrimeArithmeticSequence :=
  (a : ℕ)  -- First term
  (d : ℕ)  -- Common difference
  (h1 : 0 < d)  -- Ensure the sequence is increasing
  (h2 : ∀ i : Fin 5, Prime (a + i.val * d))  -- All 5 terms are prime

/-- The fifth term of a prime arithmetic sequence -/
def fifthTerm (seq : PrimeArithmeticSequence) : ℕ :=
  seq.a + 4 * seq.d

theorem smallest_fifth_prime_term :
  (∃ seq : PrimeArithmeticSequence, fifthTerm seq = 29) ∧
  (∀ seq : PrimeArithmeticSequence, 29 ≤ fifthTerm seq) :=
sorry

end NUMINAMATH_CALUDE_smallest_fifth_prime_term_l835_83515


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_l835_83555

theorem pure_imaginary_complex (a : ℝ) : 
  (a - (17 : ℂ) / (4 - Complex.I)).im = 0 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_l835_83555


namespace NUMINAMATH_CALUDE_set_intersection_union_theorem_l835_83507

def A : Set ℝ := {x | 2*x - x^2 ≤ x}
def B : Set ℝ := {x | x/(1-x) ≤ x/(1-x)}
def C (a b : ℝ) : Set ℝ := {x | a*x^2 + x + b < 0}

theorem set_intersection_union_theorem (a b : ℝ) :
  (A ∪ B) ∩ (C a b) = ∅ ∧ (A ∪ B) ∪ (C a b) = Set.univ →
  a = -1/3 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_union_theorem_l835_83507


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l835_83548

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Define the solution set of f(x) ≤ 0
def solution_set (a : ℝ) : Set ℝ := {x | f a x ≤ 0}

-- Theorem 1
theorem solution_set_of_inequality (a : ℝ) :
  solution_set a = Set.Icc 1 2 →
  {x : ℝ | f a x ≥ 1 - x^2} = Set.Iic (1/2) ∪ Set.Ici 1 :=
sorry

-- Theorem 2
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*a*(x-1) + 4) →
  a ∈ Set.Iic (1/3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l835_83548


namespace NUMINAMATH_CALUDE_quadratic_inequality_l835_83593

/-- Given a quadratic function f(x) = x^2 + bx + c, if f(-1) = f(3), then f(1) < c < f(3) -/
theorem quadratic_inequality (b c : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  f (-1) = f 3 → f 1 < c ∧ c < f 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l835_83593


namespace NUMINAMATH_CALUDE_six_steps_position_l835_83541

/-- Given a number line with equally spaced markings where 8 steps cover 48 units,
    prove that 6 steps from 0 reach position 36. -/
theorem six_steps_position (total_distance : ℕ) (total_steps : ℕ) (steps : ℕ) :
  total_distance = 48 →
  total_steps = 8 →
  steps = 6 →
  (total_distance / total_steps) * steps = 36 := by
  sorry

end NUMINAMATH_CALUDE_six_steps_position_l835_83541


namespace NUMINAMATH_CALUDE_tan_beta_value_l835_83543

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = -2) 
  (h2 : Real.tan (α + β) = 1) : 
  Real.tan β = -3 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l835_83543


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l835_83521

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x > 1/2 → 1/x < 2) ∧
  (∃ x : ℝ, 1/x < 2 ∧ x ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l835_83521


namespace NUMINAMATH_CALUDE_power_function_through_point_l835_83596

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f (-2) = -8) : 
  f 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l835_83596


namespace NUMINAMATH_CALUDE_doll_count_difference_l835_83501

/-- The number of dolls Geraldine has -/
def geraldine_dolls : ℝ := 2186.25

/-- The number of dolls Jazmin has -/
def jazmin_dolls : ℝ := 1209.73

/-- The number of dolls Felicia has -/
def felicia_dolls : ℝ := 1530.48

/-- The difference between Geraldine's dolls and the sum of Jazmin's and Felicia's dolls -/
def doll_difference : ℝ := geraldine_dolls - (jazmin_dolls + felicia_dolls)

theorem doll_count_difference : doll_difference = -553.96 := by
  sorry

end NUMINAMATH_CALUDE_doll_count_difference_l835_83501


namespace NUMINAMATH_CALUDE_prom_dancers_l835_83522

theorem prom_dancers (total_kids : ℕ) (slow_dancers : ℕ) 
  (h_total : total_kids = 140)
  (h_dancers : (total_kids : ℚ) / 4 = total_kids / 4)
  (h_slow : slow_dancers = 25)
  (h_ratio : ∃ (x : ℕ), x > 0 ∧ slow_dancers = 5 * x ∧ (total_kids / 4 : ℚ) = 10 * x) :
  (total_kids / 4 : ℚ) - slow_dancers = 0 :=
sorry

end NUMINAMATH_CALUDE_prom_dancers_l835_83522


namespace NUMINAMATH_CALUDE_chips_sales_third_fourth_week_l835_83508

/-- Proves that the number of bags of chips sold in each of the third and fourth week is 20 --/
theorem chips_sales_third_fourth_week :
  let total_sales : ℕ := 100
  let first_week_sales : ℕ := 15
  let second_week_sales : ℕ := 3 * first_week_sales
  let remaining_sales : ℕ := total_sales - (first_week_sales + second_week_sales)
  let third_fourth_week_sales : ℕ := remaining_sales / 2
  third_fourth_week_sales = 20 := by
  sorry

end NUMINAMATH_CALUDE_chips_sales_third_fourth_week_l835_83508


namespace NUMINAMATH_CALUDE_recipe_total_cups_l835_83537

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients used in a recipe given the ratio and cups of sugar -/
def total_cups (ratio : RecipeRatio) (sugar_cups : ℕ) : ℕ :=
  let part_size := sugar_cups / ratio.sugar
  part_size * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for a recipe with ratio 1:8:5 and 10 cups of sugar, the total cups used is 28 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := ⟨1, 8, 5⟩
  let sugar_cups : ℕ := 10
  total_cups ratio sugar_cups = 28 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l835_83537


namespace NUMINAMATH_CALUDE_cube_root_problem_l835_83572

theorem cube_root_problem (a : ℕ) (h : a^3 = 21 * 25 * 315 * 7) : a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l835_83572


namespace NUMINAMATH_CALUDE_money_ratio_problem_l835_83546

theorem money_ratio_problem (a b : ℕ) (ha : a = 800) (hb : b = 500) : 
  (a : ℚ) / b = 8 / 5 ∧ 
  ((a - 50 : ℚ) / (b + 100) = 5 / 4) := by
sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l835_83546


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l835_83518

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, n > 0 ∧ 
    is_perfect_square n ∧
    is_divisible_by n 2 ∧
    is_divisible_by n 3 ∧
    is_divisible_by n 5 ∧
    (∀ m : ℕ, m > 0 ∧ 
      is_perfect_square m ∧
      is_divisible_by m 2 ∧
      is_divisible_by m 3 ∧
      is_divisible_by m 5 →
      n ≤ m) ∧
    n = 900 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l835_83518


namespace NUMINAMATH_CALUDE_money_distribution_l835_83510

theorem money_distribution (total : ℝ) (p q r : ℝ) : 
  total = 4000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 1600 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l835_83510


namespace NUMINAMATH_CALUDE_units_digit_17_2011_l835_83524

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the property that powers of 17 have the same units digit as powers of 7
axiom units_digit_17_7 (n : ℕ) : unitsDigit (17^n) = unitsDigit (7^n)

-- Define the cycle of units digits for powers of 7
def sevenPowerCycle : List ℕ := [7, 9, 3, 1]

-- Theorem stating that the units digit of 17^2011 is 3
theorem units_digit_17_2011 : unitsDigit (17^2011) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_2011_l835_83524


namespace NUMINAMATH_CALUDE_common_tangent_range_l835_83587

/-- Given two curves y = x^2 - 1 and y = a ln x - 1, where a is a positive real number,
    if there exists a common tangent line to both curves, then 0 < a ≤ 2e. -/
theorem common_tangent_range (a : ℝ) (h_pos : a > 0) :
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
    (2 * x₁ : ℝ) = a / x₂ ∧
    x₁^2 + 1 = a + 1 - a * Real.log x₂) →
  0 < a ∧ a ≤ 2 * Real.exp 1 :=
by sorry


end NUMINAMATH_CALUDE_common_tangent_range_l835_83587


namespace NUMINAMATH_CALUDE_total_distance_traveled_l835_83520

/-- Proves that the total distance traveled is 900 kilometers given the specified conditions -/
theorem total_distance_traveled (D : ℝ) : 
  (D / 3 : ℝ) + (2 / 3 * 360 : ℝ) + 360 = D → D = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l835_83520


namespace NUMINAMATH_CALUDE_triangle_ef_length_l835_83578

/-- Given a triangle DEF with the specified conditions, prove that EF = 3 -/
theorem triangle_ef_length (D E F : ℝ) (h1 : Real.cos (2 * D - E) + Real.sin (D + E) = 2) (h2 : DE = 6) : EF = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ef_length_l835_83578
