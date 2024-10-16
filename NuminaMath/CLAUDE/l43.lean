import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l43_4384

theorem simplify_expression (x y : ℝ) : (x - y)^3 / (x - y)^2 * (y - x) = -(x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l43_4384


namespace NUMINAMATH_CALUDE_probability_triangle_or_square_l43_4390

theorem probability_triangle_or_square (total_figures : ℕ) 
  (triangle_count : ℕ) (square_count : ℕ) :
  total_figures = 10 →
  triangle_count = 3 →
  square_count = 4 →
  (triangle_count + square_count : ℚ) / total_figures = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_triangle_or_square_l43_4390


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l43_4354

def f (a b x : ℝ) : ℝ := (x + a) * abs (x + b)

theorem necessary_not_sufficient_condition :
  (∀ x, f a b x = -f a b (-x)) → a = b ∧
  ∃ a b, a = b ∧ ∃ x, f a b x ≠ -f a b (-x) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l43_4354


namespace NUMINAMATH_CALUDE_tan_150_degrees_l43_4399

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l43_4399


namespace NUMINAMATH_CALUDE_problem_statement_l43_4335

-- Define the function f and its derivative g
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

-- Define the conditions
axiom f_diff : ∀ x, HasDerivAt f (g x) x
axiom f_even : ∀ x, f (3/2 - 2*x) = f (3/2 + 2*x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- State the theorem to be proved
theorem problem_statement :
  (f (-1) = f 4) ∧ (g (-1/2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l43_4335


namespace NUMINAMATH_CALUDE_helen_cookies_l43_4398

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 435

/-- The number of cookies Helen baked this morning -/
def cookies_today : ℕ := 139

/-- The total number of cookies Helen baked -/
def total_cookies : ℕ := cookies_yesterday + cookies_today

/-- Theorem stating that the total number of cookies Helen baked is 574 -/
theorem helen_cookies : total_cookies = 574 := by sorry

end NUMINAMATH_CALUDE_helen_cookies_l43_4398


namespace NUMINAMATH_CALUDE_tenth_equation_right_side_l43_4331

/-- The sum of the first n natural numbers -/
def sum_of_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of cubes of the first n natural numbers -/
def sum_of_cubes (n : ℕ) : ℕ := (sum_of_n n) ^ 2

theorem tenth_equation_right_side :
  sum_of_cubes 10 = 55^2 := by sorry

end NUMINAMATH_CALUDE_tenth_equation_right_side_l43_4331


namespace NUMINAMATH_CALUDE_parabola_properties_l43_4347

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Theorem statement
theorem parabola_properties :
  ∃ (a : ℝ), 
    (parabola a 1 = 2) ∧ 
    (a = 2) ∧ 
    (∀ (x y : ℝ), y = -1/8 ↔ y = -(1 / (4 * a)) ∧ parabola a x = y + 1/(4*a)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l43_4347


namespace NUMINAMATH_CALUDE_beetle_journey_l43_4321

/-- Represents the beetle's movements in centimeters -/
def beetle_movements : List ℝ := [10, -9, 8, -6, 7.5, -6, 8, -7]

/-- Time taken per centimeter in seconds -/
def time_per_cm : ℝ := 2

/-- Calculates the final position of the beetle relative to the starting point -/
def final_position (movements : List ℝ) : ℝ :=
  movements.sum

/-- Calculates the total distance traveled by the beetle -/
def total_distance (movements : List ℝ) : ℝ :=
  movements.map abs |>.sum

/-- Calculates the total time taken for the journey -/
def total_time (movements : List ℝ) (time_per_cm : ℝ) : ℝ :=
  (total_distance movements) * time_per_cm

theorem beetle_journey :
  final_position beetle_movements = 5.5 ∧
  total_time beetle_movements time_per_cm = 123 := by
  sorry

#eval final_position beetle_movements
#eval total_time beetle_movements time_per_cm

end NUMINAMATH_CALUDE_beetle_journey_l43_4321


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l43_4391

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l43_4391


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_l43_4367

theorem smallest_two_digit_multiple : ∃ n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ k : ℕ, n = 30 * k + 2) ∧
  (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ (∃ j : ℕ, m = 30 * j + 2) → m ≥ n) ∧
  n = 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_l43_4367


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l43_4368

/-- Represents the number of books Robert can read in a given time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (pages_per_hour * available_hours) / pages_per_book

/-- Theorem stating that Robert can read 2 books in 6 hours -/
theorem robert_reading_capacity : books_read 90 270 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l43_4368


namespace NUMINAMATH_CALUDE_point_distance_to_y_axis_l43_4386

theorem point_distance_to_y_axis (a : ℝ) : 
  (a + 3 > 0) →  -- Point is in the first quadrant (x-coordinate is positive)
  (a > 0) →      -- Point is in the first quadrant (y-coordinate is positive)
  (a + 3 = 5) →  -- Distance to y-axis is 5
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_point_distance_to_y_axis_l43_4386


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l43_4355

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (3 * i + 1) / (1 - i)
  Complex.im z = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l43_4355


namespace NUMINAMATH_CALUDE_dogs_in_garden_l43_4376

/-- The number of dogs in a garden with ducks and a specific number of feet. -/
def num_dogs (total_feet : ℕ) (num_ducks : ℕ) (feet_per_dog : ℕ) (feet_per_duck : ℕ) : ℕ :=
  (total_feet - num_ducks * feet_per_duck) / feet_per_dog

/-- Theorem stating that under the given conditions, there are 6 dogs in the garden. -/
theorem dogs_in_garden : num_dogs 28 2 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_dogs_in_garden_l43_4376


namespace NUMINAMATH_CALUDE_power_mod_seven_l43_4320

theorem power_mod_seven : 3^255 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_power_mod_seven_l43_4320


namespace NUMINAMATH_CALUDE_initial_cards_count_l43_4304

/-- The number of cards Jennifer had initially -/
def initial_cards : ℕ := sorry

/-- The number of cards eaten by the hippopotamus -/
def eaten_cards : ℕ := 61

/-- The number of cards remaining after some were eaten -/
def remaining_cards : ℕ := 11

/-- Theorem stating that the initial number of cards is 72 -/
theorem initial_cards_count : initial_cards = 72 := by sorry

end NUMINAMATH_CALUDE_initial_cards_count_l43_4304


namespace NUMINAMATH_CALUDE_vector_sum_zero_implies_parallel_l43_4360

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_sum_zero_implies_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a + b = 0 → parallel a b) ∧ ¬(parallel a b → a + b = 0) := by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_implies_parallel_l43_4360


namespace NUMINAMATH_CALUDE_f_at_seven_l43_4317

/-- The polynomial f(x) = 7x^5 + 12x^4 - 5x^3 - 6x^2 + 3x - 5 -/
def f (x : ℝ) : ℝ := 7*x^5 + 12*x^4 - 5*x^3 - 6*x^2 + 3*x - 5

/-- Theorem stating that f(7) = 144468 -/
theorem f_at_seven : f 7 = 144468 := by
  sorry

end NUMINAMATH_CALUDE_f_at_seven_l43_4317


namespace NUMINAMATH_CALUDE_product_equals_eight_l43_4334

theorem product_equals_eight (x : ℝ) (hx : x ≠ 0) : 
  ∃ y : ℝ, x * y = 8 ∧ y = 8 / x := by sorry

end NUMINAMATH_CALUDE_product_equals_eight_l43_4334


namespace NUMINAMATH_CALUDE_number_of_terms_S_9891_1989_l43_4324

/-- Elementary symmetric expression -/
def S (k : ℕ) (n : ℕ) : ℕ := Nat.choose k n

/-- The number of terms in S_{9891}(1989) -/
theorem number_of_terms_S_9891_1989 : S 9891 1989 = Nat.choose 9891 1989 := by
  sorry

end NUMINAMATH_CALUDE_number_of_terms_S_9891_1989_l43_4324


namespace NUMINAMATH_CALUDE_basketball_tournament_matches_l43_4393

/-- The number of matches in a round-robin tournament with n teams -/
def roundRobinMatches (n : ℕ) : ℕ := n.choose 2

/-- The total number of matches played in the basketball tournament -/
def totalMatches (groups numTeams : ℕ) : ℕ :=
  groups * roundRobinMatches numTeams + roundRobinMatches groups

theorem basketball_tournament_matches :
  totalMatches 5 6 = 85 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tournament_matches_l43_4393


namespace NUMINAMATH_CALUDE_min_omega_value_l43_4333

/-- Given a function f(x) = 2 sin(ωx) with a minimum value of 2 on the interval [-π/3, π/4],
    the minimum value of ω is 3/2 -/
theorem min_omega_value (ω : ℝ) (h : ∀ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω * x) ≥ 2) :
  ω ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_omega_value_l43_4333


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l43_4337

theorem x_squared_minus_y_squared (x y : ℝ) 
  (eq1 : x + y = 4) 
  (eq2 : 2 * x - 2 * y = 1) : 
  x^2 - y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l43_4337


namespace NUMINAMATH_CALUDE_man_mass_from_boat_displacement_l43_4396

/-- Calculates the mass of a man based on the displacement of a boat -/
theorem man_mass_from_boat_displacement (boat_length boat_breadth additional_depth water_density : ℝ) 
  (h1 : boat_length = 4)
  (h2 : boat_breadth = 2)
  (h3 : additional_depth = 0.01)
  (h4 : water_density = 1000) : 
  boat_length * boat_breadth * additional_depth * water_density = 80 := by
  sorry

end NUMINAMATH_CALUDE_man_mass_from_boat_displacement_l43_4396


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l43_4362

theorem roots_sum_of_squares (r s : ℝ) : 
  r^2 - 5*r + 3 = 0 → s^2 - 5*s + 3 = 0 → r^2 + s^2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l43_4362


namespace NUMINAMATH_CALUDE_correct_quotient_l43_4319

theorem correct_quotient (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 35) : D / 21 = 20 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_l43_4319


namespace NUMINAMATH_CALUDE_women_count_at_gathering_l43_4374

/-- Represents a social gathering where men and women dance. -/
structure SocialGathering where
  men : ℕ
  women : ℕ
  manDances : ℕ
  womanDances : ℕ

/-- The number of women at the gathering is correct if the total number of dances
    from men's perspective equals the total number of dances from women's perspective. -/
def isCorrectWomenCount (g : SocialGathering) : Prop :=
  g.men * g.manDances = g.women * g.womanDances

/-- Theorem stating that in a gathering with 15 men, where each man dances with 4 women
    and each woman dances with 3 men, there are 20 women. -/
theorem women_count_at_gathering :
  ∀ g : SocialGathering,
    g.men = 15 →
    g.manDances = 4 →
    g.womanDances = 3 →
    isCorrectWomenCount g →
    g.women = 20 := by
  sorry

end NUMINAMATH_CALUDE_women_count_at_gathering_l43_4374


namespace NUMINAMATH_CALUDE_min_value_theorem_l43_4369

theorem min_value_theorem (a b : ℝ) (ha : a > 0) 
  (h : ∀ x > 0, (a * x - 2) * (-x^2 - b * x + 4) ≤ 0) : 
  ∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ ∀ b, b + 3 / a ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l43_4369


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l43_4359

theorem no_positive_integer_solution :
  ¬ ∃ (a b : ℕ+), 4 * (a^2 + a) = b^2 + b := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l43_4359


namespace NUMINAMATH_CALUDE_servant_worked_nine_months_l43_4348

/-- Represents the salary and employment duration of a servant --/
structure ServantSalary where
  yearly_cash : ℕ  -- Yearly cash salary in Rupees
  turban_price : ℕ  -- Price of the turban in Rupees
  leaving_cash : ℕ  -- Cash received when leaving in Rupees
  months_worked : ℚ  -- Number of months worked

/-- Calculates the number of months a servant worked based on the given salary structure --/
def calculate_months_worked (s : ServantSalary) : ℚ :=
  let total_yearly_salary : ℚ := s.yearly_cash + s.turban_price
  let monthly_salary : ℚ := total_yearly_salary / 12
  let total_received : ℚ := s.leaving_cash + s.turban_price
  total_received / monthly_salary

/-- Theorem stating that the servant worked for approximately 9 months --/
theorem servant_worked_nine_months (s : ServantSalary) 
  (h1 : s.yearly_cash = 90)
  (h2 : s.turban_price = 70)
  (h3 : s.leaving_cash = 50) :
  ∃ ε > 0, |calculate_months_worked s - 9| < ε := by
  sorry

#eval calculate_months_worked { yearly_cash := 90, turban_price := 70, leaving_cash := 50, months_worked := 0 }

end NUMINAMATH_CALUDE_servant_worked_nine_months_l43_4348


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l43_4345

/-- A geometric sequence with a_1 = 9 and a_5 = a_3 * a_4^2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 
    a 1 = 9 ∧ 
    (∀ n : ℕ, a (n + 1) = a n * q) ∧
    a 5 = a 3 * (a 4)^2

theorem geometric_sequence_a4 (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 4 = 1/3 ∨ a 4 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l43_4345


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l43_4395

/-- Calculate the gain percent from a scooter sale -/
theorem scooter_gain_percent 
  (purchase_price : ℝ) 
  (repair_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 800)
  (h2 : repair_cost = 200)
  (h3 : selling_price = 1200) : 
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l43_4395


namespace NUMINAMATH_CALUDE_T_bounds_not_in_T_l43_4330

-- Define the set T
def T : Set ℝ := {y | ∃ x : ℝ, x ≠ 1 ∧ y = (3*x + 4)/(x - 1)}

-- State the theorem
theorem T_bounds_not_in_T :
  (∃ M : ℝ, IsLUB T M ∧ M = 3) ∧
  (∀ m : ℝ, ¬IsGLB T m) ∧
  3 ∉ T ∧
  (∀ y : ℝ, y ∈ T → y < 3) :=
sorry

end NUMINAMATH_CALUDE_T_bounds_not_in_T_l43_4330


namespace NUMINAMATH_CALUDE_unique_solution_exists_l43_4338

theorem unique_solution_exists (m n : ℕ) : 
  (∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + m * b = n ∧ a + b = m * c) ↔ 
  (m > 1 ∧ (n - 1) % (m - 1) = 0 ∧ ¬∃k, n = m ^ k) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l43_4338


namespace NUMINAMATH_CALUDE_midline_length_l43_4379

/-- A trapezoid with perpendicular diagonals -/
structure PerpendicularDiagonalTrapezoid where
  /-- Length of one diagonal -/
  diagonal1 : ℝ
  /-- Angle between the other diagonal and the base (in radians) -/
  angle_with_base : ℝ
  /-- The diagonals are perpendicular -/
  perpendicular_diagonals : True
  /-- One diagonal is 6 units long -/
  diagonal1_length : diagonal1 = 6
  /-- The other diagonal forms a 30° angle with the base -/
  angle_30_degrees : angle_with_base = π / 6

/-- The midline of a trapezoid -/
def midline (t : PerpendicularDiagonalTrapezoid) : ℝ :=
  sorry

theorem midline_length (t : PerpendicularDiagonalTrapezoid) :
  midline t = 6 := by
  sorry

end NUMINAMATH_CALUDE_midline_length_l43_4379


namespace NUMINAMATH_CALUDE_specific_cistern_wet_area_l43_4351

/-- Calculates the total wet surface area of a rectangular cistern -/
def cisternWetArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the wet surface area of a specific cistern -/
theorem specific_cistern_wet_area :
  cisternWetArea 6 4 1.25 = 49 := by
  sorry

end NUMINAMATH_CALUDE_specific_cistern_wet_area_l43_4351


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l43_4325

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 1024 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l43_4325


namespace NUMINAMATH_CALUDE_original_houses_count_l43_4329

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 97741

/-- The current total number of houses in Lincoln County -/
def current_houses : ℕ := 118558

/-- The original number of houses in Lincoln County -/
def original_houses : ℕ := current_houses - houses_built

theorem original_houses_count : original_houses = 20817 := by
  sorry

end NUMINAMATH_CALUDE_original_houses_count_l43_4329


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l43_4300

theorem no_solution_for_equation :
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l43_4300


namespace NUMINAMATH_CALUDE_hot_dog_price_is_two_l43_4308

/-- Calculates the price of a single hot dog given the hourly sales rate, operating hours, and total sales. -/
def hot_dog_price (hourly_rate : ℕ) (hours : ℕ) (total_sales : ℕ) : ℚ :=
  total_sales / (hourly_rate * hours)

/-- Theorem stating that the price of each hot dog is $2 under given conditions. -/
theorem hot_dog_price_is_two :
  hot_dog_price 10 10 200 = 2 := by
  sorry

#eval hot_dog_price 10 10 200

end NUMINAMATH_CALUDE_hot_dog_price_is_two_l43_4308


namespace NUMINAMATH_CALUDE_cost_per_mile_calculation_l43_4314

/-- Calculates the cost per mile for a car rental --/
theorem cost_per_mile_calculation
  (daily_rental_fee : ℚ)
  (daily_budget : ℚ)
  (distance : ℚ)
  (h1 : daily_rental_fee = 30)
  (h2 : daily_budget = 76)
  (h3 : distance = 200)
  : (daily_budget - daily_rental_fee) / distance = 23 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_mile_calculation_l43_4314


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l43_4327

theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  (4 * (a / 4) = 6 * (b / 6)) → a / b = 1 := by
sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l43_4327


namespace NUMINAMATH_CALUDE_jimmy_win_probability_remainder_mod_1000_l43_4380

/-- Probability of rolling an odd number on a single die -/
def prob_odd_single : ℚ := 3/4

/-- Probability of Jimmy winning a single game -/
def prob_jimmy_win : ℚ := 1 - prob_odd_single^2

/-- Probability of Jimmy winning exactly k out of n games -/
def prob_jimmy_win_k_of_n (k n : ℕ) : ℚ :=
  Nat.choose n k * prob_jimmy_win^k * (1 - prob_jimmy_win)^(n - k)

/-- Probability of Jimmy winning 3 games before Jacob wins 3 games -/
def prob_jimmy_wins_3_first : ℚ :=
  prob_jimmy_win_k_of_n 3 3 +
  prob_jimmy_win_k_of_n 3 4 +
  prob_jimmy_win_k_of_n 3 5

theorem jimmy_win_probability :
  prob_jimmy_wins_3_first = 201341 / 2^19 :=
sorry

theorem remainder_mod_1000 :
  (201341 : ℤ) + 19 ≡ 360 [ZMOD 1000] :=
sorry

end NUMINAMATH_CALUDE_jimmy_win_probability_remainder_mod_1000_l43_4380


namespace NUMINAMATH_CALUDE_locus_of_fourth_vertex_l43_4323

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle with center and radius -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Checks if a point lies on a circle -/
def lies_on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Represents a rectangle by its vertices -/
structure Rectangle :=
  (A B C D : Point)

theorem locus_of_fourth_vertex 
  (O : Point) (r R : ℝ) (hr : 0 < r) (hR : r < R)
  (c1 : Circle) (c2 : Circle) (rect : Rectangle)
  (hc1 : c1 = ⟨O, r⟩) (hc2 : c2 = ⟨O, R⟩)
  (hA : lies_on_circle rect.A c2 ∨ lies_on_circle rect.A c1)
  (hB : lies_on_circle rect.B c2 ∨ lies_on_circle rect.B c1)
  (hD : lies_on_circle rect.D c2 ∨ lies_on_circle rect.D c1) :
  lies_on_circle rect.C c1 ∨ lies_on_circle rect.C c2 ∨
  (lies_on_circle rect.C c1 ∧ 
   (rect.C.x - O.x)^2 + (rect.C.y - O.y)^2 + 
   (rect.B.x - O.x)^2 + (rect.B.y - O.y)^2 = 2 * R^2) :=
sorry

end NUMINAMATH_CALUDE_locus_of_fourth_vertex_l43_4323


namespace NUMINAMATH_CALUDE_bobs_roommates_l43_4303

theorem bobs_roommates (john_roommates : ℕ) (h1 : john_roommates = 25) :
  ∃ (bob_roommates : ℕ), john_roommates = 2 * bob_roommates + 5 → bob_roommates = 10 := by
  sorry

end NUMINAMATH_CALUDE_bobs_roommates_l43_4303


namespace NUMINAMATH_CALUDE_mathematics_arrangements_l43_4366

def word : String := "MATHEMATICS"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowel_count (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

def consonant_count (s : String) : Nat :=
  s.length - vowel_count s

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def multiset_permutations (total : Nat) (duplicates : List Nat) : Nat :=
  factorial total / (duplicates.map factorial |>.prod)

theorem mathematics_arrangements :
  let vowels := vowel_count word
  let consonants := consonant_count word
  let vowel_arrangements := multiset_permutations vowels [2]
  let consonant_arrangements := multiset_permutations consonants [2, 2]
  vowel_arrangements * consonant_arrangements = 15120 := by
  sorry

end NUMINAMATH_CALUDE_mathematics_arrangements_l43_4366


namespace NUMINAMATH_CALUDE_initial_mean_equals_correct_mean_l43_4307

/-- Proves that the initial mean is equal to the correct mean when one value is incorrectly copied --/
theorem initial_mean_equals_correct_mean (n : ℕ) (correct_value incorrect_value : ℝ) (correct_mean : ℝ) :
  n = 25 →
  correct_value = 165 →
  incorrect_value = 130 →
  correct_mean = 191.4 →
  (n * correct_mean - correct_value + incorrect_value) / n = correct_mean := by
  sorry

#check initial_mean_equals_correct_mean

end NUMINAMATH_CALUDE_initial_mean_equals_correct_mean_l43_4307


namespace NUMINAMATH_CALUDE_total_blue_balloons_l43_4394

theorem total_blue_balloons (joan_balloons sally_balloons jessica_balloons : ℕ) 
  (h1 : joan_balloons = 9)
  (h2 : sally_balloons = 5)
  (h3 : jessica_balloons = 2) :
  joan_balloons + sally_balloons + jessica_balloons = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l43_4394


namespace NUMINAMATH_CALUDE_ball_distribution_problem_l43_4389

/-- The number of ways to distribute n distinct objects into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where the first box is not empty -/
def distributeWithFirstBoxNonEmpty (n k : ℕ) : ℕ :=
  distribute n k - distribute n (k - 1)

/-- The problem statement -/
theorem ball_distribution_problem :
  distributeWithFirstBoxNonEmpty 3 4 = 37 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_problem_l43_4389


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l43_4350

theorem students_in_both_clubs 
  (total_students : ℕ) 
  (drama_students : ℕ) 
  (science_students : ℕ) 
  (students_in_either : ℕ) 
  (h1 : total_students = 320)
  (h2 : drama_students = 90)
  (h3 : science_students = 140)
  (h4 : students_in_either = 200) :
  drama_students + science_students - students_in_either = 30 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l43_4350


namespace NUMINAMATH_CALUDE_binomial_expectation_and_variance_l43_4301

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_expectation_and_variance :
  ∀ ξ : BinomialRV, ξ.n = 10 ∧ ξ.p = 0.6 → 
  expected_value ξ = 6 ∧ variance ξ = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_and_variance_l43_4301


namespace NUMINAMATH_CALUDE_B_wins_4_probability_C_wins_3_probability_l43_4343

-- Define the players
inductive Player : Type
| A : Player
| B : Player
| C : Player

-- Define the win probabilities
def winProb (winner loser : Player) : ℝ :=
  match winner, loser with
  | Player.A, Player.B => 0.4
  | Player.B, Player.C => 0.5
  | Player.C, Player.A => 0.6
  | _, _ => 0 -- For other combinations, set probability to 0

-- Define the probability of B winning 4 consecutive matches
def prob_B_wins_4 : ℝ :=
  (1 - winProb Player.A Player.B) * 
  (winProb Player.B Player.C) * 
  (1 - winProb Player.A Player.B) * 
  (winProb Player.B Player.C)

-- Define the probability of C winning 3 consecutive matches
def prob_C_wins_3 : ℝ :=
  ((1 - winProb Player.A Player.B) * (1 - winProb Player.B Player.C) * 
   (winProb Player.C Player.A) * (1 - winProb Player.B Player.C)) +
  ((winProb Player.A Player.B) * (winProb Player.C Player.A) * 
   (1 - winProb Player.B Player.C) * (winProb Player.C Player.A))

-- Theorem statements
theorem B_wins_4_probability : prob_B_wins_4 = 0.09 := by sorry

theorem C_wins_3_probability : prob_C_wins_3 = 0.162 := by sorry

end NUMINAMATH_CALUDE_B_wins_4_probability_C_wins_3_probability_l43_4343


namespace NUMINAMATH_CALUDE_first_fund_profit_percentage_l43_4306

/-- Proves that the profit percentage of the first mutual fund is approximately 2.82% given the specified conditions --/
theorem first_fund_profit_percentage 
  (total_investment : ℝ) 
  (investment_higher_profit : ℝ) 
  (second_fund_profit : ℝ) 
  (total_profit : ℝ) 
  (h1 : total_investment = 1900)
  (h2 : investment_higher_profit = 1700)
  (h3 : second_fund_profit = 0.02)
  (h4 : total_profit = 52)
  : ∃ (first_fund_profit : ℝ), 
    (first_fund_profit * investment_higher_profit + 
     second_fund_profit * (total_investment - investment_higher_profit) = total_profit) ∧
    (abs (first_fund_profit - 0.0282) < 0.0001) :=
by sorry

end NUMINAMATH_CALUDE_first_fund_profit_percentage_l43_4306


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l43_4358

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 1} = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l43_4358


namespace NUMINAMATH_CALUDE_three_fourths_of_hundred_l43_4357

theorem three_fourths_of_hundred : (3 / 4 : ℚ) * 100 = 75 := by sorry

end NUMINAMATH_CALUDE_three_fourths_of_hundred_l43_4357


namespace NUMINAMATH_CALUDE_system_a_solution_system_b_solutions_l43_4397

-- Part (a)
theorem system_a_solution (x y : ℝ) : 
  x^2 - 3*x*y - 4*y^2 = 0 ∧ x^3 + y^3 = 65 → (x = 4 ∧ y = 1) :=
sorry

-- Part (b)
theorem system_b_solutions (x y : ℝ) :
  x^2 + 2*y^2 = 17 ∧ 2*x*y - x^2 = 3 →
  ((x = 3 ∧ y = 2) ∨ 
   (x = -3 ∧ y = -2) ∨ 
   (x = Real.sqrt 3 / 3 ∧ y = 5 * Real.sqrt 3 / 3) ∨ 
   (x = -Real.sqrt 3 / 3 ∧ y = -5 * Real.sqrt 3 / 3)) :=
sorry

end NUMINAMATH_CALUDE_system_a_solution_system_b_solutions_l43_4397


namespace NUMINAMATH_CALUDE_magician_trick_min_digits_l43_4371

/-- The minimum number of digits required for the magician's trick -/
def min_digits : ℕ := 101

/-- The number of possible two-digit combinations -/
def two_digit_combinations (n : ℕ) : ℕ := (n - 1) * (10^(n - 2))

/-- The total number of possible arrangements -/
def total_arrangements (n : ℕ) : ℕ := 10^n

/-- Theorem stating that 101 is the minimum number of digits required for the magician's trick -/
theorem magician_trick_min_digits :
  (∀ n : ℕ, n ≥ min_digits → two_digit_combinations n ≥ total_arrangements n) ∧
  (∀ n : ℕ, n < min_digits → two_digit_combinations n < total_arrangements n) :=
sorry

end NUMINAMATH_CALUDE_magician_trick_min_digits_l43_4371


namespace NUMINAMATH_CALUDE_polygon_sides_l43_4326

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 + 360 = 1260) →
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l43_4326


namespace NUMINAMATH_CALUDE_correct_distribution_l43_4310

/-- Represents the distribution of sampled students across three camps -/
structure CampDistribution where
  camp1 : Nat
  camp2 : Nat
  camp3 : Nat

/-- Parameters for the systematic sampling -/
structure SamplingParams where
  totalStudents : Nat
  sampleSize : Nat
  startNumber : Nat

/-- Function to perform systematic sampling and calculate camp distribution -/
def systematicSampling (params : SamplingParams) : CampDistribution :=
  sorry

/-- Theorem stating the correct distribution for the given problem -/
theorem correct_distribution :
  let params : SamplingParams := {
    totalStudents := 300,
    sampleSize := 20,
    startNumber := 3
  }
  let result : CampDistribution := systematicSampling params
  result.camp1 = 14 ∧ result.camp2 = 3 ∧ result.camp3 = 3 :=
sorry

end NUMINAMATH_CALUDE_correct_distribution_l43_4310


namespace NUMINAMATH_CALUDE_notebook_cost_l43_4328

/-- The cost of a notebook and a pen given two equations -/
theorem notebook_cost (n p : ℚ) 
  (eq1 : 3 * n + 4 * p = 3.75)
  (eq2 : 5 * n + 2 * p = 3.05) :
  n = 0.3357 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l43_4328


namespace NUMINAMATH_CALUDE_haley_marbles_division_l43_4349

/-- Given a number of marbles and a number of boys, calculate the number of marbles each boy receives when divided equally. -/
def marblesPerBoy (totalMarbles : ℕ) (numBoys : ℕ) : ℕ :=
  totalMarbles / numBoys

/-- Theorem stating that when 35 marbles are divided equally among 5 boys, each boy receives 7 marbles. -/
theorem haley_marbles_division :
  marblesPerBoy 35 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_haley_marbles_division_l43_4349


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_5_l43_4385

theorem x_plus_2y_equals_5 (x y : ℝ) 
  (h1 : (x + y) / 3 = 1) 
  (h2 : 2 * x + y = 4) : 
  x + 2 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_5_l43_4385


namespace NUMINAMATH_CALUDE_events_independent_l43_4311

/-- Represents the outcome of a single coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Represents the outcome of tossing a coin twice -/
def DoubleToss := CoinToss × CoinToss

/-- Event A: the first toss is heads -/
def event_A (toss : DoubleToss) : Prop :=
  toss.1 = CoinToss.Heads

/-- Event B: the second toss is tails -/
def event_B (toss : DoubleToss) : Prop :=
  toss.2 = CoinToss.Tails

/-- The probability of an event occurring -/
def probability (event : DoubleToss → Prop) : ℝ :=
  sorry

/-- Theorem: Events A and B are mutually independent -/
theorem events_independent :
  probability (fun toss ↦ event_A toss ∧ event_B toss) =
  probability event_A * probability event_B :=
sorry

end NUMINAMATH_CALUDE_events_independent_l43_4311


namespace NUMINAMATH_CALUDE_apples_handed_out_to_students_l43_4312

/-- Given a cafeteria with apples, prove the number of apples handed out to students. -/
theorem apples_handed_out_to_students 
  (initial_apples : ℕ) 
  (apples_per_pie : ℕ) 
  (pies_made : ℕ) 
  (h1 : initial_apples = 51)
  (h2 : apples_per_pie = 5)
  (h3 : pies_made = 2) :
  initial_apples - (apples_per_pie * pies_made) = 41 :=
by sorry

end NUMINAMATH_CALUDE_apples_handed_out_to_students_l43_4312


namespace NUMINAMATH_CALUDE_sugar_sold_is_two_kilograms_l43_4322

/-- The number of sugar packets sold per week -/
def packets_per_week : ℕ := 20

/-- The amount of sugar in grams per packet -/
def grams_per_packet : ℕ := 100

/-- Conversion factor from grams to kilograms -/
def grams_per_kilogram : ℕ := 1000

/-- The amount of sugar sold per week in kilograms -/
def sugar_sold_per_week : ℚ :=
  (packets_per_week * grams_per_packet : ℚ) / grams_per_kilogram

theorem sugar_sold_is_two_kilograms :
  sugar_sold_per_week = 2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_sold_is_two_kilograms_l43_4322


namespace NUMINAMATH_CALUDE_total_days_2005_to_2010_l43_4382

def is_leap_year (year : ℕ) : Bool := year = 2008

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def year_range : List ℕ := [2005, 2006, 2007, 2008, 2009, 2010]

theorem total_days_2005_to_2010 :
  (year_range.map days_in_year).sum = 2191 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2005_to_2010_l43_4382


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l43_4318

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with a₂ = 2 and S₄ = 9, a₆ = 4 -/
theorem arithmetic_sequence_a6 (seq : ArithmeticSequence) 
    (h1 : seq.a 2 = 2) 
    (h2 : seq.S 4 = 9) : 
  seq.a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l43_4318


namespace NUMINAMATH_CALUDE_total_sales_amount_theorem_l43_4336

def weight_deviations : List ℤ := [-4, -1, -2, 2, 3, 4, 7, 1]
def qualification_criterion : ℤ := 4
def price_per_bag : ℚ := 86/10

def is_qualified (deviation : ℤ) : Bool :=
  deviation.natAbs ≤ qualification_criterion

theorem total_sales_amount_theorem :
  (weight_deviations.filter is_qualified).length * price_per_bag = 602/10 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_amount_theorem_l43_4336


namespace NUMINAMATH_CALUDE_fourth_guard_runs_150_meters_l43_4364

/-- The length of the rectangle in meters -/
def length : ℝ := 200

/-- The width of the rectangle in meters -/
def width : ℝ := 300

/-- The perimeter of the rectangle in meters -/
def perimeter : ℝ := 2 * (length + width)

/-- The total distance run by three guards in meters -/
def three_guards_distance : ℝ := 850

/-- The distance run by the fourth guard in meters -/
def fourth_guard_distance : ℝ := perimeter - three_guards_distance

theorem fourth_guard_runs_150_meters :
  fourth_guard_distance = 150 := by sorry

end NUMINAMATH_CALUDE_fourth_guard_runs_150_meters_l43_4364


namespace NUMINAMATH_CALUDE_oplus_one_three_l43_4342

def oplus (x y : ℤ) : ℤ := -3 * x + 4 * y

theorem oplus_one_three : oplus 1 3 = 9 := by sorry

end NUMINAMATH_CALUDE_oplus_one_three_l43_4342


namespace NUMINAMATH_CALUDE_red_area_after_four_changes_l43_4381

/-- Represents the fraction of red area remaining after one execution of the process -/
def remaining_fraction : ℚ := 8 / 9

/-- The number of times the process is executed -/
def process_iterations : ℕ := 4

/-- Calculates the fraction of the original area that remains red after n iterations -/
def red_area_fraction (n : ℕ) : ℚ := remaining_fraction ^ n

theorem red_area_after_four_changes :
  red_area_fraction process_iterations = 4096 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_red_area_after_four_changes_l43_4381


namespace NUMINAMATH_CALUDE_total_pears_l43_4302

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17

theorem total_pears : alyssa_pears + nancy_pears = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_l43_4302


namespace NUMINAMATH_CALUDE_selection_plans_l43_4315

theorem selection_plans (n m : ℕ) (h1 : n = 6) (h2 : m = 3) : 
  (n.choose m) * m.factorial = 120 := by
  sorry

end NUMINAMATH_CALUDE_selection_plans_l43_4315


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_half_ninths_l43_4340

theorem smallest_integer_greater_than_half_ninths : ∀ n : ℤ, (1/2 : ℚ) < (n : ℚ)/9 ↔ n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_half_ninths_l43_4340


namespace NUMINAMATH_CALUDE_inequality_proof_l43_4344

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (a + 2*b + c)^2 / (2*b^2 + (c + a)^2) +
  (a + b + 2*c)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l43_4344


namespace NUMINAMATH_CALUDE_samantha_born_in_1975_l43_4339

-- Define the year of the first AMC 8
def first_amc8_year : ℕ := 1983

-- Define Samantha's age when she took the seventh AMC 8
def samantha_age_seventh_amc8 : ℕ := 14

-- Define the number of years between first and seventh AMC 8
def years_between_first_and_seventh : ℕ := 6

-- Define the year Samantha took the seventh AMC 8
def samantha_seventh_amc8_year : ℕ := first_amc8_year + years_between_first_and_seventh

-- Define Samantha's birth year
def samantha_birth_year : ℕ := samantha_seventh_amc8_year - samantha_age_seventh_amc8

-- Theorem to prove
theorem samantha_born_in_1975 : samantha_birth_year = 1975 := by
  sorry

end NUMINAMATH_CALUDE_samantha_born_in_1975_l43_4339


namespace NUMINAMATH_CALUDE_airline_capacity_example_l43_4309

/-- Calculates the total number of passengers an airline can accommodate daily --/
def airline_capacity (num_airplanes : ℕ) (rows_per_airplane : ℕ) (seats_per_row : ℕ) (flights_per_day : ℕ) : ℕ :=
  num_airplanes * rows_per_airplane * seats_per_row * flights_per_day

/-- Theorem: An airline with 5 airplanes, 20 rows per airplane, 7 seats per row, and 2 flights per day can accommodate 1400 passengers daily --/
theorem airline_capacity_example : airline_capacity 5 20 7 2 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_airline_capacity_example_l43_4309


namespace NUMINAMATH_CALUDE_sale_price_comparison_l43_4352

theorem sale_price_comparison (x : ℝ) (h : x > 0) : x * 1.3 * 0.85 > x * 1.1 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_comparison_l43_4352


namespace NUMINAMATH_CALUDE_f_is_quadratic_l43_4383

/-- Definition of a one-variable quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 2 * (x - x^2) - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l43_4383


namespace NUMINAMATH_CALUDE_definite_integral_x_squared_plus_sin_l43_4372

open Real MeasureTheory

theorem definite_integral_x_squared_plus_sin : 
  ∫ x in (-1)..1, (x^2 + Real.sin x) = 2/3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_x_squared_plus_sin_l43_4372


namespace NUMINAMATH_CALUDE_arithmetic_computation_l43_4346

theorem arithmetic_computation : -5 * (-6) - 2 * (-3 * (-7) + (-8)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l43_4346


namespace NUMINAMATH_CALUDE_max_value_of_f_l43_4377

open Real

-- Define the function
def f (x : ℝ) : ℝ := x * (3 - 2 * x)

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (3/2) ∧
  (∀ x, x ∈ Set.Ioo 0 (3/2) → f x ≤ f c) ∧
  f c = 9/8 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l43_4377


namespace NUMINAMATH_CALUDE_roots_equation_l43_4365

open Real

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := x^2 - 2 * cos θ * x + 1

theorem roots_equation (θ : ℝ) (α : ℝ) 
  (h1 : f θ (sin α) = 1/4 + cos θ) 
  (h2 : f θ (cos α) = 1/4 + cos θ) : 
  (tan α)^2 + 1 / tan α = (16 + 4 * sqrt 11) / 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l43_4365


namespace NUMINAMATH_CALUDE_max_lessons_l43_4332

/-- Represents the number of shirts the teacher has. -/
def s : ℕ := sorry

/-- Represents the number of pairs of pants the teacher has. -/
def p : ℕ := sorry

/-- Represents the number of pairs of shoes the teacher has. -/
def b : ℕ := sorry

/-- Represents the number of jackets the teacher has. -/
def jackets : ℕ := 2

/-- Represents the total number of possible lessons. -/
def total_lessons : ℕ := 2 * s * p * b

/-- States that one more shirt would allow 36 more lessons. -/
axiom shirt_condition : 2 * (s + 1) * p * b = total_lessons + 36

/-- States that one more pair of pants would allow 72 more lessons. -/
axiom pants_condition : 2 * s * (p + 1) * b = total_lessons + 72

/-- States that one more pair of shoes would allow 54 more lessons. -/
axiom shoes_condition : 2 * s * p * (b + 1) = total_lessons + 54

/-- Theorem stating the maximum number of lessons the teacher could have conducted. -/
theorem max_lessons : total_lessons = 216 := by sorry

end NUMINAMATH_CALUDE_max_lessons_l43_4332


namespace NUMINAMATH_CALUDE_total_length_is_16cm_l43_4370

/-- Represents the dimensions of a rectangle in centimeters -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the segments removed from the rectangle -/
structure RemovedSegments where
  long_side : ℝ
  short_side_ends : ℝ

/-- Represents the split in the remaining short side -/
structure SplitSegment where
  distance_from_middle : ℝ

/-- Calculates the total length of segments after modifications -/
def total_length_after_modifications (rect : Rectangle) (removed : RemovedSegments) (split : SplitSegment) : ℝ :=
  let remaining_long_side := rect.length - removed.long_side
  let remaining_short_side := rect.width - 2 * removed.short_side_ends
  let split_segment := min split.distance_from_middle (remaining_short_side / 2)
  remaining_long_side + remaining_short_side + 2 * removed.short_side_ends

/-- Theorem stating that the total length of segments after modifications is 16 cm -/
theorem total_length_is_16cm (rect : Rectangle) (removed : RemovedSegments) (split : SplitSegment)
    (h1 : rect.length = 10)
    (h2 : rect.width = 5)
    (h3 : removed.long_side = 3)
    (h4 : removed.short_side_ends = 2)
    (h5 : split.distance_from_middle = 1) :
  total_length_after_modifications rect removed split = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_length_is_16cm_l43_4370


namespace NUMINAMATH_CALUDE_inverse_of_42_mod_43_and_59_l43_4313

theorem inverse_of_42_mod_43_and_59 :
  (∃ x : ℤ, (42 * x) % 43 = 1) ∧ (∃ y : ℤ, (42 * y) % 59 = 1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_42_mod_43_and_59_l43_4313


namespace NUMINAMATH_CALUDE_infinite_52_divisible_cells_l43_4361

/-- Represents a position in the grid -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- The value at a node given its position in the spiral -/
def spiral_value (p : Position) : ℕ := sorry

/-- The sum of values at the four corners of a cell -/
def cell_sum (p : Position) : ℕ :=
  spiral_value p + spiral_value ⟨p.x + 1, p.y⟩ + 
  spiral_value ⟨p.x + 1, p.y + 1⟩ + spiral_value ⟨p.x, p.y + 1⟩

/-- Predicate for whether a number is divisible by 52 -/
def divisible_by_52 (n : ℕ) : Prop := n % 52 = 0

/-- The main theorem to be proved -/
theorem infinite_52_divisible_cells :
  ∀ n : ℕ, ∃ p : Position, p.x ≥ n ∧ p.y ≥ n ∧ divisible_by_52 (cell_sum p) :=
sorry

end NUMINAMATH_CALUDE_infinite_52_divisible_cells_l43_4361


namespace NUMINAMATH_CALUDE_power_product_equality_l43_4375

theorem power_product_equality : (-2/3)^2023 * (3/2)^2024 = -3/2 := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l43_4375


namespace NUMINAMATH_CALUDE_percent_of_y_l43_4387

theorem percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 10 + (3 * y) / 10) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l43_4387


namespace NUMINAMATH_CALUDE_remainder_23_pow_2003_mod_7_l43_4392

theorem remainder_23_pow_2003_mod_7 : 23^2003 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_23_pow_2003_mod_7_l43_4392


namespace NUMINAMATH_CALUDE_linear_decreasing_slope_l43_4305

/-- For a linear function y = (m-2)x + 1, if y is decreasing as x increases, then m < 2. -/
theorem linear_decreasing_slope (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((m - 2) * x₁ + 1) > ((m - 2) * x₂ + 1)) →
  m < 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_decreasing_slope_l43_4305


namespace NUMINAMATH_CALUDE_zack_group_size_l43_4363

/-- Proves that Zack tutors students in groups of 10, given the problem conditions -/
theorem zack_group_size :
  ∀ (x : ℕ),
  (∃ (n : ℕ), x * n = 70) →  -- Zack tutors 70 students in total
  (∃ (m : ℕ), 10 * m = 70) →  -- Karen tutors 70 students in total
  x = 10 := by sorry

end NUMINAMATH_CALUDE_zack_group_size_l43_4363


namespace NUMINAMATH_CALUDE_range_of_m_l43_4388

-- Define P and q as functions of x and m
def P (x : ℝ) : Prop := |4 - x| / 3 ≤ 2

def q (x m : ℝ) : Prop := (x + m - 1) * (x - m - 1) ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(P x) → ¬(q x m)) →
  (∃ x, P x ∧ ¬(q x m)) →
  m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l43_4388


namespace NUMINAMATH_CALUDE_percy_swimming_weeks_l43_4316

/-- Represents Percy's swimming schedule and calculates the number of weeks to swim a given total hours -/
def swimming_schedule (weekday_hours_per_day : ℕ) (weekday_days : ℕ) (weekend_hours : ℕ) (total_hours : ℕ) : ℕ :=
  let hours_per_week := weekday_hours_per_day * weekday_days + weekend_hours
  total_hours / hours_per_week

/-- Proves that Percy's swimming schedule over 52 hours covers 4 weeks -/
theorem percy_swimming_weeks : swimming_schedule 2 5 3 52 = 4 := by
  sorry

#eval swimming_schedule 2 5 3 52

end NUMINAMATH_CALUDE_percy_swimming_weeks_l43_4316


namespace NUMINAMATH_CALUDE_polynomial_factorization_l43_4356

theorem polynomial_factorization (x y : ℝ) : x^3 * y - 4 * x * y^3 = x * y * (x + 2 * y) * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l43_4356


namespace NUMINAMATH_CALUDE_jeans_savings_l43_4373

/-- Calculates the total amount saved on a purchase with multiple discounts and taxes -/
def calculateSavings (originalPrice : ℝ) (saleDiscount : ℝ) (couponDiscount : ℝ) 
                     (creditCardDiscount : ℝ) (rebateDiscount : ℝ) (salesTax : ℝ) : ℝ :=
  let priceAfterSale := originalPrice * (1 - saleDiscount)
  let priceAfterCoupon := priceAfterSale - couponDiscount
  let priceAfterCreditCard := priceAfterCoupon * (1 - creditCardDiscount)
  let priceBeforeRebate := priceAfterCreditCard
  let taxAmount := priceBeforeRebate * salesTax
  let finalPrice := (priceBeforeRebate - priceBeforeRebate * rebateDiscount) + taxAmount
  originalPrice - finalPrice

theorem jeans_savings :
  calculateSavings 125 0.20 10 0.10 0.05 0.08 = 41.57 := by
  sorry

end NUMINAMATH_CALUDE_jeans_savings_l43_4373


namespace NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l43_4353

/-- Represents the number of tickets Tom has -/
structure TomTickets where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Conversion rates between ticket types -/
def yellowToRed : ℕ := 10
def redToBlue : ℕ := 10

/-- Number of yellow tickets needed to win a Bible -/
def yellowToWin : ℕ := 10

/-- Tom's current tickets -/
def tomCurrentTickets : TomTickets :=
  { yellow := 8, red := 3, blue := 7 }

/-- Calculate the total number of blue tickets Tom has -/
def totalBlueTickets (t : TomTickets) : ℕ :=
  t.yellow * yellowToRed * redToBlue + t.red * redToBlue + t.blue

/-- Calculate the number of blue tickets needed to win -/
def blueTicketsToWin : ℕ := yellowToWin * yellowToRed * redToBlue

/-- Theorem: Tom needs 163 more blue tickets to win a Bible -/
theorem tom_needs_163_blue_tickets :
  blueTicketsToWin - totalBlueTickets tomCurrentTickets = 163 := by
  sorry


end NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l43_4353


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l43_4341

theorem geometric_sequence_middle_term (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- a, b, c form a geometric sequence
  a = 5 + 2 * Real.sqrt 6 →
  c = 5 - 2 * Real.sqrt 6 →
  b = 1 ∨ b = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l43_4341


namespace NUMINAMATH_CALUDE_susan_board_game_l43_4378

theorem susan_board_game (total_spaces : ℕ) (first_move : ℕ) (second_move : ℕ) (third_move : ℕ) (spaces_to_win : ℕ) :
  total_spaces = 48 →
  first_move = 8 →
  second_move = 2 →
  third_move = 6 →
  spaces_to_win = 37 →
  ∃ (spaces_moved_back : ℕ),
    first_move + second_move + third_move - spaces_moved_back = total_spaces - spaces_to_win ∧
    spaces_moved_back = 6 :=
by sorry

end NUMINAMATH_CALUDE_susan_board_game_l43_4378
