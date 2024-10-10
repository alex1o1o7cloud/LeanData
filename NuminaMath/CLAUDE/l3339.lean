import Mathlib

namespace cube_zero_of_fourth_power_zero_l3339_333919

theorem cube_zero_of_fourth_power_zero (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : A ^ 3 = 0 := by sorry

end cube_zero_of_fourth_power_zero_l3339_333919


namespace regular_polygon_with_160_degree_angles_has_18_sides_l3339_333973

/-- A regular polygon with interior angles measuring 160° has 18 sides. -/
theorem regular_polygon_with_160_degree_angles_has_18_sides :
  ∀ n : ℕ, n ≥ 3 →
  (∀ θ : ℝ, θ = 160 → (n : ℝ) * θ = (n - 2 : ℝ) * 180) →
  n = 18 := by
  sorry

end regular_polygon_with_160_degree_angles_has_18_sides_l3339_333973


namespace infinite_solutions_l3339_333913

/-- The equation (x-1)^2 + (x+1)^2 = y^2 + 1 -/
def is_solution (x y : ℕ) : Prop :=
  (x - 1)^2 + (x + 1)^2 = y^2 + 1

/-- The transformation function -/
def transform (x y : ℕ) : ℕ × ℕ :=
  (3*x + 2*y, 4*x + 3*y)

theorem infinite_solutions :
  (is_solution 0 1) ∧
  (is_solution 2 3) ∧
  (∀ x y : ℕ, is_solution x y → is_solution (transform x y).1 (transform x y).2) →
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ, is_solution (f n).1 (f n).2 ∧ f n ≠ f (n+1) :=
by sorry

end infinite_solutions_l3339_333913


namespace parabola_focus_l3339_333900

/-- The focus of a parabola y = ax^2 (a ≠ 0) is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let parabola := {(x, y) : ℝ × ℝ | y = a * x^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (0, 1 / (4 * a)) :=
sorry

end parabola_focus_l3339_333900


namespace find_b_value_l3339_333970

theorem find_b_value (circle_sum : ℕ) (total_sum : ℕ) (d : ℕ) :
  circle_sum = 21 * 5 ∧
  total_sum = 69 ∧
  d + 5 + 9 = 21 →
  ∃ b : ℕ, b = 10 ∧ circle_sum - (2 + 8 + 9 + b + d) = total_sum :=
by sorry

end find_b_value_l3339_333970


namespace unique_solution_xy_equation_l3339_333955

theorem unique_solution_xy_equation :
  ∃! (x y : ℕ), x < y ∧ x^y = y^x :=
by sorry

end unique_solution_xy_equation_l3339_333955


namespace sum_of_xyz_equals_six_l3339_333956

theorem sum_of_xyz_equals_six (a b : ℝ) (x y z : ℤ) : 
  a^2 = 9/36 → 
  b^2 = (1 + Real.sqrt 3)^2 / 8 → 
  a < 0 → 
  b > 0 → 
  (a - b)^2 = (x : ℝ) * Real.sqrt y / z → 
  x + y + z = 6 := by sorry

end sum_of_xyz_equals_six_l3339_333956


namespace running_race_participants_l3339_333996

theorem running_race_participants (first_grade : ℕ) (second_grade : ℕ) : 
  first_grade = 8 →
  second_grade = 5 * first_grade →
  first_grade + second_grade = 48 := by
  sorry

end running_race_participants_l3339_333996


namespace weekly_payment_problem_l3339_333917

/-- The weekly payment problem -/
theorem weekly_payment_problem (payment_B : ℝ) (payment_ratio : ℝ) : 
  payment_B = 180 →
  payment_ratio = 1.5 →
  payment_B + payment_ratio * payment_B = 450 := by
  sorry

end weekly_payment_problem_l3339_333917


namespace alex_has_48_shells_l3339_333995

/-- The number of seashells in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of seashells Mimi picked up -/
def mimi_dozens : ℕ := 2

/-- The number of seashells Mimi picked up -/
def mimi_shells : ℕ := mimi_dozens * dozen

/-- The number of seashells Kyle found -/
def kyle_shells : ℕ := 2 * mimi_shells

/-- The number of seashells Leigh grabbed -/
def leigh_shells : ℕ := kyle_shells / 3

/-- The number of seashells Alex unearthed -/
def alex_shells : ℕ := 3 * leigh_shells

/-- Theorem stating that Alex had 48 seashells -/
theorem alex_has_48_shells : alex_shells = 48 := by
  sorry

end alex_has_48_shells_l3339_333995


namespace fixed_point_of_exponential_function_l3339_333986

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 4
  f 1 = 5 := by
sorry

end fixed_point_of_exponential_function_l3339_333986


namespace vectors_coplanar_iff_x_eq_five_l3339_333954

/-- Given vectors a, b, and c in ℝ³, prove that they are coplanar if and only if x = 5 -/
theorem vectors_coplanar_iff_x_eq_five (a b c : ℝ × ℝ × ℝ) :
  a = (1, -1, 3) →
  b = (-1, 4, -2) →
  c = (1, 5, x) →
  (∃ (m n : ℝ), c = m • a + n • b) ↔ x = 5 := by
  sorry

end vectors_coplanar_iff_x_eq_five_l3339_333954


namespace quadratic_coefficients_4x2_eq_3_l3339_333981

/-- Given a quadratic equation ax^2 + bx + c = 0, returns the tuple (a, b, c) -/
def quadratic_coefficients (f : ℝ → ℝ) : ℝ × ℝ × ℝ := sorry

theorem quadratic_coefficients_4x2_eq_3 :
  quadratic_coefficients (fun x => 4 * x^2 - 3) = (4, 0, -3) := by sorry

end quadratic_coefficients_4x2_eq_3_l3339_333981


namespace total_tires_changed_is_304_l3339_333930

/-- Represents the number of tires changed by Mike in a day -/
def total_tires_changed : ℕ :=
  let motorcycles := 12
  let cars := 10
  let bicycles := 8
  let trucks := 5
  let atvs := 7
  let dual_axle_trailers := 4
  let triple_axle_boat_trailers := 3
  let unicycles := 2
  let dually_pickup_trucks := 6

  let motorcycle_tires := 2
  let car_tires := 4
  let bicycle_tires := 2
  let truck_tires := 18
  let atv_tires := 4
  let dual_axle_trailer_tires := 8
  let triple_axle_boat_trailer_tires := 12
  let unicycle_tires := 1
  let dually_pickup_truck_tires := 6

  motorcycles * motorcycle_tires +
  cars * car_tires +
  bicycles * bicycle_tires +
  trucks * truck_tires +
  atvs * atv_tires +
  dual_axle_trailers * dual_axle_trailer_tires +
  triple_axle_boat_trailers * triple_axle_boat_trailer_tires +
  unicycles * unicycle_tires +
  dually_pickup_trucks * dually_pickup_truck_tires

/-- Theorem stating that the total number of tires changed by Mike in a day is 304 -/
theorem total_tires_changed_is_304 : total_tires_changed = 304 := by
  sorry

end total_tires_changed_is_304_l3339_333930


namespace prob_six_queen_ace_l3339_333944

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of cards of each rank (e.g., 6, Queen, Ace) in a standard deck -/
def CardsPerRank : ℕ := 4

/-- Probability of drawing a specific sequence of three cards from a standard deck -/
def prob_specific_sequence (deck_size : ℕ) (cards_per_rank : ℕ) : ℚ :=
  (cards_per_rank : ℚ) / deck_size *
  (cards_per_rank : ℚ) / (deck_size - 1) *
  (cards_per_rank : ℚ) / (deck_size - 2)

theorem prob_six_queen_ace :
  prob_specific_sequence StandardDeck CardsPerRank = 16 / 33150 := by
  sorry

end prob_six_queen_ace_l3339_333944


namespace percentage_commutation_l3339_333951

theorem percentage_commutation (n : ℝ) (h : 0.20 * 0.10 * n = 12) : 0.10 * 0.20 * n = 12 := by
  sorry

end percentage_commutation_l3339_333951


namespace five_sixteenths_decimal_l3339_333966

theorem five_sixteenths_decimal : (5 : ℚ) / 16 = 0.3125 := by
  sorry

end five_sixteenths_decimal_l3339_333966


namespace complex_square_simplification_l3339_333968

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 + 3 * i)^2 = 7 + 24 * i :=
by sorry

end complex_square_simplification_l3339_333968


namespace probability_two_pairs_is_5_21_l3339_333962

-- Define the total number of socks and colors
def total_socks : ℕ := 10
def num_colors : ℕ := 5
def socks_per_color : ℕ := 2
def socks_drawn : ℕ := 5

-- Define the probability function
def probability_two_pairs : ℚ :=
  let total_combinations := Nat.choose total_socks socks_drawn
  let favorable_combinations := Nat.choose num_colors 2 * Nat.choose (num_colors - 2) 1 * socks_per_color
  (favorable_combinations : ℚ) / total_combinations

-- Theorem statement
theorem probability_two_pairs_is_5_21 : 
  probability_two_pairs = 5 / 21 := by sorry

end probability_two_pairs_is_5_21_l3339_333962


namespace slowest_pump_time_l3339_333971

/-- Three pumps with rates in ratio 2:3:4 fill a pool in 6 hours. The slowest pump fills it in 27 hours. -/
theorem slowest_pump_time (pool_volume : ℝ) (h : pool_volume > 0) : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧  -- Pump rates are positive
    r₂ = (3/2) * r₁ ∧           -- Ratio of rates
    r₃ = 2 * r₁ ∧               -- Ratio of rates
    (r₁ + r₂ + r₃) * 6 = pool_volume ∧  -- All pumps fill the pool in 6 hours
    r₁ * 27 = pool_volume       -- Slowest pump fills the pool in 27 hours
  := by sorry

end slowest_pump_time_l3339_333971


namespace log_equation_solution_l3339_333978

theorem log_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.log x + Real.log (x + 4) = Real.log (2 * x + 8) :=
by sorry

end log_equation_solution_l3339_333978


namespace second_red_ball_most_likely_l3339_333984

/-- The total number of balls in the urn -/
def total_balls : ℕ := 101

/-- The number of red balls in the urn -/
def red_balls : ℕ := 3

/-- The probability of drawing the second red ball on the kth draw -/
def prob_second_red (k : ℕ) : ℚ :=
  if 1 < k ∧ k < total_balls
  then (k - 1 : ℚ) * (total_balls - k : ℚ) / (total_balls.choose red_balls : ℚ)
  else 0

/-- The draw number that maximizes the probability of drawing the second red ball -/
def max_prob_draw : ℕ := 51

theorem second_red_ball_most_likely :
  ∀ k, prob_second_red max_prob_draw ≥ prob_second_red k :=
sorry

end second_red_ball_most_likely_l3339_333984


namespace angle_sum_is_pi_over_two_l3339_333943

theorem angle_sum_is_pi_over_two (α β : Real) : 
  (0 < α ∧ α < π / 2) →  -- α is acute
  (0 < β ∧ β < π / 2) →  -- β is acute
  3 * (Real.sin α) ^ 2 + 2 * (Real.sin β) ^ 2 = 1 →
  3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0 →
  α + 2 * β = π / 2 := by
sorry

end angle_sum_is_pi_over_two_l3339_333943


namespace isosceles_triangle_perimeter_l3339_333911

/-- An isosceles triangle with sides of 4cm and 3cm has a perimeter of either 10cm or 11cm. -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Triangle inequality
  (a = 4 ∧ b = 3) ∨ (a = 3 ∧ b = 4) →  -- Given side lengths
  ((a = b ∧ c = 3) ∨ (a = c ∧ b = 3)) →  -- Isosceles condition
  a + b + c = 10 ∨ a + b + c = 11 :=
by sorry

end isosceles_triangle_perimeter_l3339_333911


namespace kevins_phone_repair_l3339_333974

/-- Given the initial conditions of Kevin's phone repair scenario, 
    prove that the number of phones each person needs to repair is 9. -/
theorem kevins_phone_repair 
  (initial_phones : ℕ) 
  (repaired_phones : ℕ) 
  (new_phones : ℕ) 
  (h1 : initial_phones = 15)
  (h2 : repaired_phones = 3)
  (h3 : new_phones = 6) :
  (initial_phones - repaired_phones + new_phones) / 2 = 9 := by
sorry

end kevins_phone_repair_l3339_333974


namespace parabola_equation_correct_l3339_333925

/-- Represents a parabola with equation ax^2 + bx + c --/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a parabola has a vertical axis of symmetry --/
def has_vertical_axis_of_symmetry (p : Parabola) : Prop :=
  p.a ≠ 0

/-- Computes the vertex of a parabola --/
def vertex (p : Parabola) : ℚ × ℚ :=
  (- p.b / (2 * p.a), - (p.b^2 - 4*p.a*p.c) / (4 * p.a))

/-- Checks if a point lies on the parabola --/
def point_on_parabola (p : Parabola) (x y : ℚ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The main theorem --/
theorem parabola_equation_correct :
  let p : Parabola := { a := 2/9, b := -4/3, c := 0 }
  has_vertical_axis_of_symmetry p ∧
  vertex p = (3, -2) ∧
  point_on_parabola p 6 0 := by sorry

end parabola_equation_correct_l3339_333925


namespace river_crossing_trips_l3339_333934

/-- Represents the number of trips required to transport one adult across the river -/
def trips_per_adult : ℕ := 4

/-- Represents the total number of adults to be transported -/
def total_adults : ℕ := 358

/-- Calculates the total number of trips required to transport all adults -/
def total_trips : ℕ := trips_per_adult * total_adults

/-- Theorem stating that the total number of trips is 1432 -/
theorem river_crossing_trips : total_trips = 1432 := by
  sorry

end river_crossing_trips_l3339_333934


namespace complement_of_A_in_U_l3339_333936

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Finset ℕ := {2, 4, 5}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 6, 7} := by sorry

end complement_of_A_in_U_l3339_333936


namespace eve_walking_distance_l3339_333975

theorem eve_walking_distance (ran_distance : Real) (extra_distance : Real) :
  ran_distance = 0.7 ∧ extra_distance = 0.1 →
  ∃ walked_distance : Real, walked_distance = ran_distance - extra_distance ∧ walked_distance = 0.6 :=
by sorry

end eve_walking_distance_l3339_333975


namespace complex_number_equal_parts_l3339_333939

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (2 - Complex.I)
  Complex.re z = Complex.im z → a = 1/3 := by
  sorry

end complex_number_equal_parts_l3339_333939


namespace weighted_average_closer_to_larger_set_l3339_333910

theorem weighted_average_closer_to_larger_set 
  (set1 set2 : Finset ℝ) 
  (mean1 mean2 : ℝ) 
  (h_size : set1.card > set2.card) 
  (h_mean1 : mean1 = (set1.sum id) / set1.card) 
  (h_mean2 : mean2 = (set2.sum id) / set2.card) 
  (h_total_mean : (set1.sum id + set2.sum id) / (set1.card + set2.card) = 80) :
  |80 - mean1| < |80 - mean2| :=
sorry

end weighted_average_closer_to_larger_set_l3339_333910


namespace find_number_l3339_333994

theorem find_number : ∃ x : ℕ, x * 99999 = 65818408915 ∧ x = 658185 := by
  sorry

end find_number_l3339_333994


namespace exists_divisible_figure_l3339_333976

/-- A non-rectangular grid figure composed of cells -/
structure GridFigure where
  cells : ℕ
  nonRectangular : Bool

/-- Represents the ability to divide a figure into equal parts -/
def isDivisible (f : GridFigure) (n : ℕ) : Prop :=
  ∃ (k : ℕ), f.cells = n * k

/-- The main theorem stating the existence of a figure divisible by 2 to 7 -/
theorem exists_divisible_figure :
  ∃ (f : GridFigure), f.nonRectangular ∧
    (∀ n : ℕ, 2 ≤ n ∧ n ≤ 7 → isDivisible f n) :=
  sorry

end exists_divisible_figure_l3339_333976


namespace complex_power_equality_l3339_333992

theorem complex_power_equality : (1 - Complex.I) ^ (2 * Complex.I) = 2 := by
  sorry

end complex_power_equality_l3339_333992


namespace systematic_sampling_interval_l3339_333903

/-- The sampling interval for systematic sampling -/
def sampling_interval (population_size : ℕ) (sample_size : ℕ) : ℕ :=
  population_size / sample_size

/-- Theorem: The sampling interval for a population of 1000 and sample size of 20 is 50 -/
theorem systematic_sampling_interval :
  sampling_interval 1000 20 = 50 := by
  sorry

end systematic_sampling_interval_l3339_333903


namespace polynomial_division_quotient_remainder_l3339_333923

theorem polynomial_division_quotient_remainder 
  (x : ℝ) (h : x ≠ 1) : 
  ∃ (q r : ℝ), 
    x^5 + 5 = (x - 1) * q + r ∧ 
    q = x^4 + x^3 + x^2 + x + 1 ∧ 
    r = 6 :=
by sorry

end polynomial_division_quotient_remainder_l3339_333923


namespace open_box_volume_l3339_333946

/-- The volume of an open box formed by cutting squares from a rectangular sheet -/
theorem open_box_volume (sheet_length sheet_width cut_size : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_size = 7)
  (h4 : sheet_length > 2 * cut_size)
  (h5 : sheet_width > 2 * cut_size) :
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 5244 :=
by sorry

end open_box_volume_l3339_333946


namespace largest_number_with_same_quotient_and_remainder_l3339_333927

theorem largest_number_with_same_quotient_and_remainder : ∃ (n : ℕ), n = 90 ∧
  (∀ m : ℕ, m > n →
    ¬(∃ (q r : ℕ), m = 13 * q + r ∧ m = 15 * q + r ∧ r < 13 ∧ r < 15)) ∧
  (∃ (q r : ℕ), n = 13 * q + r ∧ n = 15 * q + r ∧ r < 13 ∧ r < 15) :=
by sorry

end largest_number_with_same_quotient_and_remainder_l3339_333927


namespace remaining_time_indeterminate_l3339_333972

/-- Represents the state of a math test -/
structure MathTest where
  totalProblems : ℕ
  firstInterval : ℕ
  secondInterval : ℕ
  problemsCompletedFirst : ℕ
  problemsCompletedSecond : ℕ
  problemsLeft : ℕ

/-- Theorem stating that the remaining time cannot be determined -/
theorem remaining_time_indeterminate (test : MathTest) 
  (h1 : test.totalProblems = 75)
  (h2 : test.firstInterval = 20)
  (h3 : test.secondInterval = 20)
  (h4 : test.problemsCompletedFirst = 10)
  (h5 : test.problemsCompletedSecond = 2 * test.problemsCompletedFirst)
  (h6 : test.problemsLeft = 45)
  (h7 : test.totalProblems = test.problemsCompletedFirst + test.problemsCompletedSecond + test.problemsLeft) :
  ¬∃ (remainingTime : ℕ), True := by
  sorry

#check remaining_time_indeterminate

end remaining_time_indeterminate_l3339_333972


namespace integral_f_equals_two_l3339_333924

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if -1 ≤ x ∧ x ≤ 1 then x^3 + Real.sin x
  else if 1 < x ∧ x ≤ 2 then 2
  else 0  -- We need to define f for all real numbers

-- State the theorem
theorem integral_f_equals_two : 
  ∫ x in (-1)..(2), f x = 2 := by sorry

end integral_f_equals_two_l3339_333924


namespace eggs_problem_l3339_333942

theorem eggs_problem (initial_eggs : ℕ) : 
  (initial_eggs / 2 : ℕ) - 15 = 21 → initial_eggs = 72 := by
  sorry

end eggs_problem_l3339_333942


namespace negation_proposition_l3339_333928

theorem negation_proposition :
  (∀ x : ℝ, x < 0 → x^2 ≤ 0) ↔ ¬(∃ x₀ : ℝ, x₀ < 0 ∧ x₀^2 > 0) :=
sorry

end negation_proposition_l3339_333928


namespace correct_ratio_maintenance_l3339_333980

-- Define the original recipe ratios
def flour_original : ℚ := 4
def sugar_original : ℚ := 7
def salt_original : ℚ := 2

-- Define Mary's mistake
def flour_mistake : ℚ := 2

-- Define the function to calculate additional flour needed
def additional_flour (f_orig f_mistake s_orig : ℚ) : ℚ :=
  f_orig - f_mistake

-- Define the function to calculate the difference between additional flour and salt
def flour_salt_difference (f_orig f_mistake s_orig : ℚ) : ℚ :=
  additional_flour f_orig f_mistake s_orig - 0

-- Theorem statement
theorem correct_ratio_maintenance :
  flour_salt_difference flour_original flour_mistake salt_original = 2 := by
  sorry

end correct_ratio_maintenance_l3339_333980


namespace goat_price_calculation_l3339_333948

theorem goat_price_calculation (total_cost total_hens total_goats hen_price : ℕ) 
  (h1 : total_cost = 10000)
  (h2 : total_hens = 35)
  (h3 : total_goats = 15)
  (h4 : hen_price = 125) :
  (total_cost - total_hens * hen_price) / total_goats = 375 := by
  sorry

end goat_price_calculation_l3339_333948


namespace reciprocal_of_three_halves_l3339_333997

theorem reciprocal_of_three_halves (x : ℚ) : x = 3 / 2 → 1 / x = 2 / 3 := by
  sorry

end reciprocal_of_three_halves_l3339_333997


namespace solution_set_f_leq_5_max_m_for_f_geq_quadratic_l3339_333977

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for part (I)
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part (II)
theorem max_m_for_f_geq_quadratic :
  ∃ (m : ℝ), m = 2 ∧
  (∀ x ∈ Set.Icc 0 2, f x ≥ -x^2 + 2*x + m) ∧
  (∀ m' > m, ∃ x ∈ Set.Icc 0 2, f x < -x^2 + 2*x + m') := by sorry

end solution_set_f_leq_5_max_m_for_f_geq_quadratic_l3339_333977


namespace race_distance_l3339_333908

theorem race_distance (d : ℝ) 
  (h1 : ∃ x y : ℝ, x > y ∧ d / x = (d - 25) / y)
  (h2 : ∃ y z : ℝ, y > z ∧ d / y = (d - 15) / z)
  (h3 : ∃ x z : ℝ, x > z ∧ d / x = (d - 35) / z)
  : d = 75 := by
  sorry

end race_distance_l3339_333908


namespace heat_engine_efficiencies_l3339_333957

/-- Heat engine efficiencies problem -/
theorem heat_engine_efficiencies
  (η₀ η₁ η₂ Q₁₂ Q₁₃ Q₃₄ α : ℝ)
  (h₀ : η₀ = 1 - Q₃₄ / Q₁₂)
  (h₁ : η₁ = 1 - Q₁₃ / Q₁₂)
  (h₂ : η₂ = 1 - Q₃₄ / Q₁₃)
  (h₃ : η₂ = (η₀ - η₁) / (1 - η₁))
  (h₄ : η₁ < η₀)
  (h₅ : η₂ < η₀)
  (h₆ : η₀ < 1)
  (h₇ : η₁ < 1)
  (h₈ : η₁ = (1 - 0.01 * α) * η₀) :
  η₂ = α / (100 - (100 - α) * η₀) := by
  sorry

end heat_engine_efficiencies_l3339_333957


namespace complex_magnitude_theorem_l3339_333941

theorem complex_magnitude_theorem (ω : ℂ) (h : ω = 8 + 3*I) : 
  Complex.abs (ω^2 + 6*ω + 73) = Real.sqrt 32740 := by
  sorry

end complex_magnitude_theorem_l3339_333941


namespace inequality_proof_l3339_333982

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  a^4 * b + b^4 * c + c^4 * a > a * b^4 + b * c^4 + c * a^4 := by
  sorry

end inequality_proof_l3339_333982


namespace sum_of_digits_after_addition_l3339_333915

def sum_of_digits (n : ℕ) : ℕ := sorry

def number_of_carries (a b : ℕ) : ℕ := sorry

theorem sum_of_digits_after_addition (A B : ℕ) 
  (hA : A > 0) 
  (hB : B > 0) 
  (hSumA : sum_of_digits A = 19) 
  (hSumB : sum_of_digits B = 20) 
  (hCarries : number_of_carries A B = 2) : 
  sum_of_digits (A + B) = 21 := by sorry

end sum_of_digits_after_addition_l3339_333915


namespace bullying_instances_count_l3339_333969

-- Define the given constants
def suspension_days_per_instance : ℕ := 3
def typical_person_digits : ℕ := 20
def suspension_multiplier : ℕ := 3

-- Define the total suspension days
def total_suspension_days : ℕ := suspension_multiplier * typical_person_digits

-- Define the number of bullying instances
def bullying_instances : ℕ := total_suspension_days / suspension_days_per_instance

-- Theorem statement
theorem bullying_instances_count : bullying_instances = 20 := by
  sorry

end bullying_instances_count_l3339_333969


namespace razorback_shop_profit_l3339_333999

/-- The amount the shop makes off each jersey in dollars. -/
def jersey_profit : ℕ := 34

/-- The additional cost of a t-shirt compared to a jersey in dollars. -/
def tshirt_additional_cost : ℕ := 158

/-- The amount the shop makes off each t-shirt in dollars. -/
def tshirt_profit : ℕ := jersey_profit + tshirt_additional_cost

theorem razorback_shop_profit : tshirt_profit = 192 := by
  sorry

end razorback_shop_profit_l3339_333999


namespace log_equation_solution_l3339_333987

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end log_equation_solution_l3339_333987


namespace triangle_trigonometric_identities_l3339_333905

/-- 
Given a triangle with sides a, b, c, angles α, β, γ, semi-perimeter p, inradius r, and circumradius R,
this theorem states two trigonometric identities related to the triangle.
-/
theorem triangle_trigonometric_identities 
  (a b c : ℝ) 
  (α β γ : ℝ) 
  (p r R : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = Real.pi)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_inradius : r > 0)
  (h_circumradius : R > 0) :
  (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = (p^2 - r^2 - 4*r*R) / (2*R^2) ∧
  4*R^2 * Real.cos α * Real.cos β * Real.cos γ = p^2 - (2*R + r)^2 := by
  sorry

end triangle_trigonometric_identities_l3339_333905


namespace vector_difference_magnitude_l3339_333931

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

theorem vector_difference_magnitude : ‖a - b‖ = 5 := by
  sorry

end vector_difference_magnitude_l3339_333931


namespace range_of_P_l3339_333952

theorem range_of_P (x y : ℝ) (h : x^2/3 + y^2 = 1) :
  2 ≤ |2*x + y - 4| + |4 - x - 2*y| ∧ |2*x + y - 4| + |4 - x - 2*y| ≤ 14 := by
  sorry

end range_of_P_l3339_333952


namespace stream_speed_l3339_333961

/-- Given a boat's travel times and distances, prove the speed of the stream --/
theorem stream_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) 
  (h1 : downstream_distance = 100) 
  (h2 : downstream_time = 4)
  (h3 : upstream_distance = 75)
  (h4 : upstream_time = 15) :
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = downstream_distance / downstream_time ∧
    boat_speed - stream_speed = upstream_distance / upstream_time ∧
    stream_speed = 10 := by
  sorry

#check stream_speed

end stream_speed_l3339_333961


namespace fib_100_mod_5_l3339_333965

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end fib_100_mod_5_l3339_333965


namespace square_diff_equality_l3339_333901

theorem square_diff_equality (x y A : ℝ) : 
  (2*x - y)^2 + A = (2*x + y)^2 → A = 8*x*y := by
  sorry

end square_diff_equality_l3339_333901


namespace matrix_equation_solution_l3339_333929

-- Define the matrix expression evaluation rule
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define the equation to solve
def equation (x : ℝ) : Prop :=
  matrix_value (3 * x) (x + 2) (x + 1) (2 * x) = 6

-- State the theorem
theorem matrix_equation_solution :
  ∃ x₁ x₂ : ℝ, x₁ = -2 + Real.sqrt 10 ∧ x₂ = -2 - Real.sqrt 10 ∧
  ∀ x : ℝ, equation x ↔ (x = x₁ ∨ x = x₂) :=
sorry

end matrix_equation_solution_l3339_333929


namespace expression_factorization_l3339_333909

theorem expression_factorization (x : ℝ) :
  (8 * x^6 + 36 * x^4 - 5) - (2 * x^6 - 6 * x^4 + 5) = 2 * (3 * x^6 + 21 * x^4 - 5) := by
  sorry

end expression_factorization_l3339_333909


namespace quadratic_root_sum_l3339_333990

theorem quadratic_root_sum (p q : ℝ) : 
  (∃ x : ℂ, x^2 + p*x + q = 0 ∧ x = 1 + Complex.I) → p + q = 0 := by
  sorry

end quadratic_root_sum_l3339_333990


namespace equal_roots_quadratic_l3339_333920

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + m = 0 → y = x) → 
  m = 1 := by
sorry

end equal_roots_quadratic_l3339_333920


namespace sugar_cups_correct_l3339_333993

/-- Represents the number of cups of sugar in the lemonade mixture -/
def sugar : ℕ := 28

/-- Represents the number of cups of water in the lemonade mixture -/
def water : ℕ := 56

/-- The total number of cups used in the mixture -/
def total_cups : ℕ := 84

/-- Theorem stating that the number of cups of sugar is correct given the conditions -/
theorem sugar_cups_correct :
  (sugar + water = total_cups) ∧ (2 * sugar = water) ∧ (sugar = 28) := by
  sorry

end sugar_cups_correct_l3339_333993


namespace incorrect_permutations_hello_l3339_333989

def word := "hello"

theorem incorrect_permutations_hello :
  let total_letters := word.length
  let duplicate_letters := 2  -- number of 'l's
  let total_permutations := Nat.factorial total_letters
  let unique_permutations := total_permutations / (Nat.factorial duplicate_letters)
  unique_permutations - 1 = 59 := by
  sorry

end incorrect_permutations_hello_l3339_333989


namespace some_number_less_than_two_l3339_333938

theorem some_number_less_than_two (x y : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + y < 9)
  (h6 : x = 7) : 
  y < 2 := by
sorry

end some_number_less_than_two_l3339_333938


namespace casper_candies_proof_l3339_333932

/-- The number of candies Casper initially had -/
def initial_candies : ℕ := 622

/-- The number of candies Casper gave to his brother on day 1 -/
def brother_candies : ℕ := 3

/-- The number of candies Casper gave to his sister on day 2 -/
def sister_candies : ℕ := 5

/-- The number of candies Casper gave to his friend on day 3 -/
def friend_candies : ℕ := 2

/-- The number of candies Casper had left on day 4 -/
def final_candies : ℕ := 10

theorem casper_candies_proof :
  (1 / 48 : ℚ) * initial_candies - 71 / 24 = final_candies := by
  sorry

end casper_candies_proof_l3339_333932


namespace total_spending_is_correct_l3339_333958

-- Define the structure for a week's theater visit
structure WeekVisit where
  duration : Float
  price_per_hour : Float
  discount_rate : Float
  visit_count : Nat

-- Define the list of visits for 6 weeks
def theater_visits : List WeekVisit := [
  { duration := 3, price_per_hour := 5, discount_rate := 0.2, visit_count := 1 },
  { duration := 2.5, price_per_hour := 6, discount_rate := 0.1, visit_count := 1 },
  { duration := 4, price_per_hour := 4, discount_rate := 0, visit_count := 1 },
  { duration := 3, price_per_hour := 5, discount_rate := 0.2, visit_count := 1 },
  { duration := 3.5, price_per_hour := 6, discount_rate := 0.1, visit_count := 2 },
  { duration := 2, price_per_hour := 7, discount_rate := 0, visit_count := 1 }
]

-- Define the transportation cost per visit
def transportation_cost : Float := 3

-- Calculate the total cost for a single visit
def visit_cost (visit : WeekVisit) : Float :=
  let performance_cost := visit.duration * visit.price_per_hour
  let discount := performance_cost * visit.discount_rate
  let discounted_cost := performance_cost - discount
  discounted_cost + transportation_cost

-- Calculate the total spending for all visits
def total_spending : Float :=
  theater_visits.map (fun visit => visit_cost visit * visit.visit_count.toFloat) |>.sum

-- Theorem statement
theorem total_spending_is_correct : total_spending = 126.3 := by
  sorry

end total_spending_is_correct_l3339_333958


namespace order_of_expressions_l3339_333983

theorem order_of_expressions (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  let a := Real.sqrt ((x^2 + y^2) / 2) - (x + y) / 2
  let b := (x + y) / 2 - Real.sqrt (x * y)
  let c := Real.sqrt (x * y) - 2 / (1 / x + 1 / y)
  b > a ∧ a > c := by sorry

end order_of_expressions_l3339_333983


namespace kiera_envelopes_l3339_333960

theorem kiera_envelopes (blue : ℕ) (yellow : ℕ) (green : ℕ) 
  (h1 : blue = 14)
  (h2 : yellow = blue - 6)
  (h3 : green = 3 * yellow) :
  blue + yellow + green = 46 := by
  sorry

end kiera_envelopes_l3339_333960


namespace min_stamps_for_47_cents_l3339_333991

/-- Represents the number of stamps of each denomination -/
structure StampCombination where
  threes : ℕ
  fours : ℕ
  fives : ℕ

/-- Calculates the total value of stamps in cents -/
def total_value (sc : StampCombination) : ℕ :=
  3 * sc.threes + 4 * sc.fours + 5 * sc.fives

/-- Calculates the total number of stamps -/
def total_stamps (sc : StampCombination) : ℕ :=
  sc.threes + sc.fours + sc.fives

/-- Checks if at least two types of stamps are used -/
def uses_at_least_two_types (sc : StampCombination) : Prop :=
  (sc.threes > 0 && sc.fours > 0) || (sc.threes > 0 && sc.fives > 0) || (sc.fours > 0 && sc.fives > 0)

/-- States that 10 is the minimum number of stamps needed to make 47 cents -/
theorem min_stamps_for_47_cents :
  ∃ (sc : StampCombination),
    total_value sc = 47 ∧
    uses_at_least_two_types sc ∧
    total_stamps sc = 10 ∧
    (∀ (sc' : StampCombination),
      total_value sc' = 47 →
      uses_at_least_two_types sc' →
      total_stamps sc' ≥ 10) :=
  sorry

end min_stamps_for_47_cents_l3339_333991


namespace nested_sqrt_equality_l3339_333906

theorem nested_sqrt_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15/16) := by
  sorry

end nested_sqrt_equality_l3339_333906


namespace simplify_expression_one_simplify_expression_two_l3339_333953

-- Part 1
theorem simplify_expression_one : 2 * Real.sqrt 3 * 31.5 * 612 = 6 := by sorry

-- Part 2
theorem simplify_expression_two : 
  (Real.log 3 / Real.log 4 - Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 1/4 := by sorry

end simplify_expression_one_simplify_expression_two_l3339_333953


namespace leaves_broke_after_initial_loss_l3339_333947

/-- 
Given that Ryan initially collected 89 leaves, lost 24 leaves, and now has 22 leaves left,
this theorem proves that 43 leaves broke after the initial loss of 24 leaves.
-/
theorem leaves_broke_after_initial_loss 
  (initial_leaves : ℕ) 
  (initial_loss : ℕ) 
  (final_leaves : ℕ) 
  (h1 : initial_leaves = 89)
  (h2 : initial_loss = 24)
  (h3 : final_leaves = 22) :
  initial_leaves - initial_loss - final_leaves = 43 := by
  sorry

end leaves_broke_after_initial_loss_l3339_333947


namespace heartsuit_zero_heartsuit_self_heartsuit_positive_l3339_333940

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem 1: x ♡ 0 = x^2 for all real x
theorem heartsuit_zero (x : ℝ) : heartsuit x 0 = x^2 := by sorry

-- Theorem 2: x ♡ x = 0 for all real x
theorem heartsuit_self (x : ℝ) : heartsuit x x = 0 := by sorry

-- Theorem 3: If x > y, then x ♡ y > 0 for all real x and y
theorem heartsuit_positive {x y : ℝ} (h : x > y) : heartsuit x y > 0 := by sorry

end heartsuit_zero_heartsuit_self_heartsuit_positive_l3339_333940


namespace average_daily_attendance_l3339_333926

def monday_attendance : ℕ := 10
def tuesday_attendance : ℕ := 15
def wednesday_to_friday_attendance : ℕ := 10
def total_days : ℕ := 5

def total_attendance : ℕ := 
  monday_attendance + tuesday_attendance + 3 * wednesday_to_friday_attendance

theorem average_daily_attendance : 
  total_attendance / total_days = 11 := by
  sorry

end average_daily_attendance_l3339_333926


namespace min_value_sum_reciprocals_l3339_333949

theorem min_value_sum_reciprocals (n : ℕ) (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
  (1 / (1 + a^n)) + (1 / (1 + b^n)) ≥ 1 ∧ 
  ((1 / (1 + 1^n)) + (1 / (1 + 1^n)) = 1) :=
sorry

end min_value_sum_reciprocals_l3339_333949


namespace children_at_play_l3339_333935

/-- Represents the number of children attending a play given specific conditions --/
def children_attending (adult_price child_price total_people total_revenue : ℕ) 
  (senior_citizens group_size : ℕ) : ℕ :=
  total_people - (total_revenue - child_price * (total_people - senior_citizens - group_size)) / 
    (adult_price - child_price)

/-- Theorem stating that under the given conditions, 20 children attended the play --/
theorem children_at_play : children_attending 12 6 80 840 3 15 = 20 := by
  sorry

end children_at_play_l3339_333935


namespace circle_equation_proof_l3339_333904

-- Define a circle in R^2
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define when a circle is tangent to the x-axis
def TangentToXAxis (c : Set (ℝ × ℝ)) : Prop :=
  ∃ x : ℝ, (x, 0) ∈ c ∧ ∀ y : ℝ, y ≠ 0 → (x, y) ∉ c

theorem circle_equation_proof :
  let c : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - 2)^2 = 4}
  c = Circle (0, 2) 2 ∧ TangentToXAxis c := by sorry

end circle_equation_proof_l3339_333904


namespace four_number_sequence_l3339_333937

theorem four_number_sequence (a b c d : ℝ) 
  (h1 : b^2 = a*c)
  (h2 : a*b*c = 216)
  (h3 : 2*c = b + d)
  (h4 : b + c + d = 12) :
  a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2 := by
sorry

end four_number_sequence_l3339_333937


namespace complement_of_B_in_A_l3339_333964

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 3, 5}

theorem complement_of_B_in_A : (A \ B) = {2, 4} := by sorry

end complement_of_B_in_A_l3339_333964


namespace alia_markers_l3339_333979

theorem alia_markers (steve_markers : ℕ) (austin_markers : ℕ) (alia_markers : ℕ)
  (h1 : steve_markers = 60)
  (h2 : austin_markers = steve_markers / 3)
  (h3 : alia_markers = 2 * austin_markers) :
  alia_markers = 40 := by
sorry

end alia_markers_l3339_333979


namespace square_equals_product_solution_l3339_333921

theorem square_equals_product_solution :
  ∀ a b : ℕ, a^2 = b * (b + 7) → (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end square_equals_product_solution_l3339_333921


namespace jumping_rooks_remainder_l3339_333912

/-- The number of ways to place 2n jumping rooks on an n×n chessboard 
    such that each rook attacks exactly two other rooks. -/
def f (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | 3 => 6
  | n + 1 => n.choose 2 * (2 * f n + n * f (n - 1))

/-- The main theorem stating that the number of ways to place 16 jumping rooks
    on an 8×8 chessboard, with each rook attacking exactly two others,
    when divided by 1000, gives a remainder of 530. -/
theorem jumping_rooks_remainder : f 8 % 1000 = 530 := by
  sorry


end jumping_rooks_remainder_l3339_333912


namespace infinitely_many_friendly_squares_l3339_333918

/-- A number is friendly if the set {1, 2, ..., N} can be partitioned into disjoint pairs 
    where the sum of each pair is a perfect square -/
def is_friendly (N : ℕ) : Prop :=
  ∃ (partition : List (ℕ × ℕ)), 
    (∀ (pair : ℕ × ℕ), pair ∈ partition → pair.1 ∈ Finset.range N ∧ pair.2 ∈ Finset.range N) ∧
    (∀ n ∈ Finset.range N, ∃ (pair : ℕ × ℕ), pair ∈ partition ∧ (n = pair.1 ∨ n = pair.2)) ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → ∃ k : ℕ, pair.1 + pair.2 = k^2)

/-- There are infinitely many friendly perfect squares -/
theorem infinitely_many_friendly_squares :
  ∀ (p : ℕ), p ≥ 2 → ∃ (N : ℕ), N = 2^(2*p - 3) ∧ is_friendly N ∧ ∃ (k : ℕ), N = k^2 :=
sorry

end infinitely_many_friendly_squares_l3339_333918


namespace x_power_minus_reciprocal_l3339_333907

theorem x_power_minus_reciprocal (φ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : x - 1 / x = (2 * Complex.I * Real.sin φ))
  (h4 : Odd n) :
  x^n - 1 / x^n = 2 * Complex.I^n * (Real.sin φ)^n :=
by sorry

end x_power_minus_reciprocal_l3339_333907


namespace vertex_on_x_axis_l3339_333945

/-- The parabola function -/
def f (d : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + d

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (d : ℝ) : ℝ := f d vertex_x

/-- The theorem stating that the vertex lies on the x-axis iff d = 9 -/
theorem vertex_on_x_axis (d : ℝ) : vertex_y d = 0 ↔ d = 9 := by sorry

end vertex_on_x_axis_l3339_333945


namespace expression_evaluation_l3339_333967

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  2 * x^2 + y^2 - z^2 + 3 * x * y = -6 := by
sorry

end expression_evaluation_l3339_333967


namespace field_trip_buses_l3339_333933

/-- Given a field trip scenario with vans and buses, calculate the number of buses required. -/
theorem field_trip_buses (total_people : ℕ) (num_vans : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ)
  (h1 : total_people = 180)
  (h2 : num_vans = 6)
  (h3 : people_per_van = 6)
  (h4 : people_per_bus = 18) :
  (total_people - num_vans * people_per_van) / people_per_bus = 8 :=
by sorry

end field_trip_buses_l3339_333933


namespace arithmetic_sequence_sum_l3339_333950

/-- Given an arithmetic sequence {a_n}, prove that a₃ + a₆ + a₉ = 33,
    when a₁ + a₄ + a₇ = 45 and a₂ + a₅ + a₈ = 39 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 4 + a 7 = 45 →
  a 2 + a 5 + a 8 = 39 →
  a 3 + a 6 + a 9 = 33 := by
sorry

end arithmetic_sequence_sum_l3339_333950


namespace product_of_numbers_l3339_333916

theorem product_of_numbers (x y : ℚ) : 
  (- x = 3 / 4) → (y = x - 1 / 2) → (x * y = 15 / 16) := by
  sorry

end product_of_numbers_l3339_333916


namespace investment_distribution_l3339_333988

def total_investment : ℝ := 1500
def final_amount : ℝ := 1800
def years : ℕ := 3

def interest_rate_trusty : ℝ := 0.04
def interest_rate_solid : ℝ := 0.06
def interest_rate_quick : ℝ := 0.07

def compound_factor (rate : ℝ) (years : ℕ) : ℝ :=
  (1 + rate) ^ years

theorem investment_distribution (x y : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ total_investment →
  x * compound_factor interest_rate_trusty years +
  y * compound_factor interest_rate_solid years +
  (total_investment - x - y) * compound_factor interest_rate_quick years = final_amount →
  x = 375 := by sorry

end investment_distribution_l3339_333988


namespace unique_student_count_l3339_333959

theorem unique_student_count : ∃! n : ℕ, 
  100 < n ∧ n < 200 ∧ 
  ∃ k : ℕ, n = 4 * k + 1 ∧
  ∃ m : ℕ, n = 3 * m + 2 ∧
  ∃ l : ℕ, n = 7 * l + 3 ∧
  n = 101 :=
sorry

end unique_student_count_l3339_333959


namespace infinite_special_numbers_l3339_333985

theorem infinite_special_numbers :
  ∃ (seq : ℕ → ℕ), 
    (∀ i, ∃ n, seq i = n) ∧
    (∀ i j, i < j → seq i < seq j) ∧
    (∀ i, ∀ p : ℕ, Prime p → p ∣ (seq i)^2 + 3 →
      ∃ k : ℕ, k^2 < seq i ∧ p ∣ k^2 + 3) :=
by sorry

end infinite_special_numbers_l3339_333985


namespace max_value_3m_4n_l3339_333902

theorem max_value_3m_4n (m n : ℕ) (h_sum : m * (m + 1) + n^2 ≤ 1987) (h_n_odd : Odd n) :
  3 * m + 4 * n ≤ 221 :=
sorry

end max_value_3m_4n_l3339_333902


namespace absolute_value_condition_l3339_333963

theorem absolute_value_condition (x : ℝ) :
  (∀ x, x < -2 → |x| > 2) ∧ 
  (∃ x, |x| > 2 ∧ x ≥ -2) :=
by sorry

end absolute_value_condition_l3339_333963


namespace angle_between_c_and_a_plus_b_is_zero_l3339_333922

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem angle_between_c_and_a_plus_b_is_zero
  (a b c : V)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hab : ‖a‖ = ‖b‖)
  (habgt : ‖a‖ > ‖a + b‖)
  (hc_eq : ‖c‖ = ‖a + b‖) :
  Real.arccos (inner c (a + b) / (‖c‖ * ‖a + b‖)) = 0 := by
  sorry

end angle_between_c_and_a_plus_b_is_zero_l3339_333922


namespace bob_improvement_percentage_l3339_333914

def bob_time : ℝ := 640
def sister_time : ℝ := 557

theorem bob_improvement_percentage :
  let time_difference := bob_time - sister_time
  let percentage_improvement := (time_difference / bob_time) * 100
  ∃ ε > 0, abs (percentage_improvement - 12.97) < ε :=
by sorry

end bob_improvement_percentage_l3339_333914


namespace hyperbola_asymptotes_l3339_333998

/-- The asymptotes of the hyperbola x²/9 - y²/16 = 1 are given by y = ±(4/3)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → ℝ := λ x y => x^2 / 9 - y^2 / 16 - 1
  ∃ (k : ℝ), k > 0 ∧ ∀ (x y : ℝ), h x y = 0 → y = k * x ∨ y = -k * x :=
by
  sorry

end hyperbola_asymptotes_l3339_333998
