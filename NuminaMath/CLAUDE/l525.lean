import Mathlib

namespace distance_cos80_sin80_to_cos20_sin20_l525_52524

/-- The distance between points (cos 80°, sin 80°) and (cos 20°, sin 20°) is 1. -/
theorem distance_cos80_sin80_to_cos20_sin20 : 
  let A : ℝ × ℝ := (Real.cos (80 * π / 180), Real.sin (80 * π / 180))
  let B : ℝ × ℝ := (Real.cos (20 * π / 180), Real.sin (20 * π / 180))
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1 := by
sorry

end distance_cos80_sin80_to_cos20_sin20_l525_52524


namespace pipe_fill_time_l525_52582

/-- Time for pipe to fill tank without leak -/
def T : ℝ := 5

/-- Time to fill tank with leak -/
def fill_time_with_leak : ℝ := 10

/-- Time for leak to empty full tank -/
def leak_empty_time : ℝ := 10

/-- Theorem: The pipe fills the tank in 5 hours without the leak -/
theorem pipe_fill_time :
  (1 / T - 1 / leak_empty_time = 1 / fill_time_with_leak) →
  T = 5 := by
  sorry

end pipe_fill_time_l525_52582


namespace alyssa_cut_roses_l525_52518

/-- Represents the number of roses Alyssa cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Proves that Alyssa cut 11 roses given the initial and final number of roses -/
theorem alyssa_cut_roses : roses_cut 3 14 = 11 := by
  sorry

end alyssa_cut_roses_l525_52518


namespace excellent_set_properties_l525_52562

-- Definition of an excellent set
def IsExcellentSet (M : Set ℝ) : Prop :=
  ∀ x y, x ∈ M → y ∈ M → (x + y) ∈ M ∧ (x - y) ∈ M

-- Theorem statement
theorem excellent_set_properties
  (A B : Set ℝ)
  (hA : IsExcellentSet A)
  (hB : IsExcellentSet B) :
  (IsExcellentSet (A ∩ B)) ∧
  (IsExcellentSet (A ∪ B) → (A ⊆ B ∨ B ⊆ A)) ∧
  (IsExcellentSet (A ∪ B) → IsExcellentSet (A ∩ B)) :=
by sorry

end excellent_set_properties_l525_52562


namespace football_players_count_l525_52549

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 36)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 7) :
  total - neither - (tennis - both) = 26 := by
  sorry

end football_players_count_l525_52549


namespace max_distance_line_equation_l525_52571

/-- A line in 2D space --/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Returns true if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Returns true if a point (x, y) is on the given line --/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- Returns the distance between two parallel lines --/
noncomputable def distance_between_parallel_lines (l1 l2 : Line) : ℝ :=
  sorry

/-- The theorem to be proved --/
theorem max_distance_line_equation (l1 l2 : Line) (A B : ℝ × ℝ) :
  are_parallel l1 l2 →
  point_on_line l1 A.1 A.2 →
  point_on_line l2 B.1 B.2 →
  A = (1, 3) →
  B = (2, 4) →
  (∀ l1' l2' : Line, are_parallel l1' l2' →
    point_on_line l1' A.1 A.2 →
    point_on_line l2' B.1 B.2 →
    distance_between_parallel_lines l1' l2' ≤ distance_between_parallel_lines l1 l2) →
  l1 = { slope := -1, y_intercept := 4 } :=
sorry

end max_distance_line_equation_l525_52571


namespace easier_decryption_with_more_unique_letters_l525_52597

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem easier_decryption_with_more_unique_letters 
  (word1 : String) (word2 : String) 
  (h1 : word1 = "термометр") (h2 : word2 = "ремонт") :
  (unique_letters word2).card > (unique_letters word1).card :=
by sorry

end easier_decryption_with_more_unique_letters_l525_52597


namespace root_sum_quotient_l525_52555

theorem root_sum_quotient (m₁ m₂ : ℝ) : 
  m₁^2 - 21*m₁ + 4 = 0 → 
  m₂^2 - 21*m₂ + 4 = 0 → 
  m₁ / m₂ + m₂ / m₁ = 108.25 := by
  sorry

end root_sum_quotient_l525_52555


namespace trigonometric_equality_l525_52525

theorem trigonometric_equality : 
  3.427 * Real.cos (50 * π / 180) + 
  8 * Real.cos (200 * π / 180) * Real.cos (220 * π / 180) * Real.cos (80 * π / 180) = 
  2 * Real.sin (65 * π / 180) ^ 2 := by
  sorry

end trigonometric_equality_l525_52525


namespace variance_implies_stability_l525_52529

-- Define a structure for a data set
structure DataSet where
  variance : ℝ
  stability : ℝ

-- Define a relation for comparing stability
def more_stable (a b : DataSet) : Prop :=
  a.stability > b.stability

-- Theorem statement
theorem variance_implies_stability (a b : DataSet) 
  (h : a.variance < b.variance) : more_stable a b :=
sorry

end variance_implies_stability_l525_52529


namespace points_above_line_t_range_l525_52590

def P (t : ℝ) : ℝ × ℝ := (1, t)
def Q (t : ℝ) : ℝ × ℝ := (t^2, t - 1)

def above_line (p : ℝ × ℝ) : Prop := p.1 + p.2 - 1 > 0

theorem points_above_line_t_range :
  ∀ t : ℝ, (above_line (P t) ∧ above_line (Q t)) ↔ t > 1 := by
sorry

end points_above_line_t_range_l525_52590


namespace power_three_mod_ten_l525_52586

theorem power_three_mod_ten : 3^24 % 10 = 1 := by
  sorry

end power_three_mod_ten_l525_52586


namespace emma_yield_calculation_l525_52557

/-- The annual yield percentage of Emma's investment -/
def emma_yield : ℝ := 18.33

/-- Emma's investment amount -/
def emma_investment : ℝ := 300

/-- Briana's investment amount -/
def briana_investment : ℝ := 500

/-- Briana's annual yield percentage -/
def briana_yield : ℝ := 10

/-- The difference in return-on-investment after 2 years -/
def roi_difference : ℝ := 10

/-- The number of years for the investment -/
def years : ℝ := 2

theorem emma_yield_calculation :
  emma_investment * (emma_yield / 100) * years - 
  briana_investment * (briana_yield / 100) * years = roi_difference :=
sorry

end emma_yield_calculation_l525_52557


namespace zachary_sold_40_games_l525_52576

/-- Represents the sale of video games by three friends -/
structure VideoGameSale where
  /-- Amount of money Zachary received -/
  zachary_amount : ℝ
  /-- Price per game Zachary sold -/
  price_per_game : ℝ
  /-- Total amount received by all three friends -/
  total_amount : ℝ

/-- Theorem stating that Zachary sold 40 games given the conditions -/
theorem zachary_sold_40_games (sale : VideoGameSale)
  (h1 : sale.price_per_game = 5)
  (h2 : sale.zachary_amount + (sale.zachary_amount * 1.3) + (sale.zachary_amount * 1.3 + 50) = sale.total_amount)
  (h3 : sale.total_amount = 770) :
  sale.zachary_amount / sale.price_per_game = 40 := by
sorry


end zachary_sold_40_games_l525_52576


namespace find_number_l525_52573

theorem find_number : ∃ X : ℝ, (50 : ℝ) = 0.2 * X + 47 ∧ X = 15 := by sorry

end find_number_l525_52573


namespace fraction_sum_zero_l525_52546

theorem fraction_sum_zero (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h : (a - d) / (b - c) + (b - d) / (c - a) + (c - d) / (a - b) = 0) :
  (a + d) / (b - c)^3 + (b + d) / (c - a)^3 + (c + d) / (a - b)^3 = 0 := by
  sorry

end fraction_sum_zero_l525_52546


namespace max_removable_edges_in_complete_graph_l525_52579

theorem max_removable_edges_in_complete_graph :
  ∀ (n : ℕ), n = 30 →
  ∃ (k : ℕ), k = 406 ∧
  (((n * (n - 1)) / 2) - k = n - 1) ∧
  k = ((n * (n - 1)) / 2) - (n - 1) :=
by sorry

end max_removable_edges_in_complete_graph_l525_52579


namespace chocolate_bars_left_chocolate_problem_l525_52535

theorem chocolate_bars_left (initial_bars : ℕ) 
  (thomas_and_friends : ℕ) (piper_reduction : ℕ) 
  (friend_return : ℕ) : ℕ :=
  let thomas_take := initial_bars / 4
  let friend_take := thomas_take / thomas_and_friends
  let total_taken := thomas_take - friend_return
  let piper_take := total_taken - piper_reduction
  initial_bars - total_taken - piper_take

theorem chocolate_problem : 
  chocolate_bars_left 200 5 5 5 = 115 := by
  sorry

end chocolate_bars_left_chocolate_problem_l525_52535


namespace expression_simplification_l525_52523

theorem expression_simplification :
  let a := Real.sqrt 2
  let b := Real.sqrt 3
  b > a ∧ (8 : ℝ) ^ (1/3) = 2 →
  |a - b| + (8 : ℝ) ^ (1/3) - a * (a - 1) = b := by
sorry

end expression_simplification_l525_52523


namespace quadratic_negative_value_l525_52568

theorem quadratic_negative_value (a b c : ℝ) :
  (∃ x : ℝ, x^2 + b*x + c = 0) →
  (∃ x : ℝ, a*x^2 + x + c = 0) →
  (∃ x : ℝ, a*x^2 + b*x + 1 = 0) →
  (∃ x : ℝ, a*x^2 + b*x + c < 0) :=
by sorry

end quadratic_negative_value_l525_52568


namespace max_value_polynomial_l525_52519

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  ∃ (max : ℝ), max = 72.25 ∧ 
  ∀ (a b : ℝ), a + b = 5 → 
    a^5*b + a^4*b + a^3*b + a*b + a*b^2 + a*b^3 + a*b^5 ≤ max :=
by sorry

end max_value_polynomial_l525_52519


namespace total_weight_is_56_7_l525_52556

/-- The total weight of five plastic rings in grams -/
def total_weight_in_grams : ℝ :=
  let orange_weight := 0.08333333333333333
  let purple_weight := 0.3333333333333333
  let white_weight := 0.4166666666666667
  let blue_weight := 0.5416666666666666
  let red_weight := 0.625
  let conversion_factor := 28.35
  (orange_weight + purple_weight + white_weight + blue_weight + red_weight) * conversion_factor

/-- Theorem stating that the total weight of the five plastic rings is 56.7 grams -/
theorem total_weight_is_56_7 : total_weight_in_grams = 56.7 := by
  sorry

end total_weight_is_56_7_l525_52556


namespace regular_price_is_15_l525_52599

-- Define the variables
def num_shirts : ℕ := 20
def discount_rate : ℚ := 0.2
def tax_rate : ℚ := 0.1
def total_paid : ℚ := 264

-- Define the theorem
theorem regular_price_is_15 :
  ∃ (regular_price : ℚ),
    regular_price * num_shirts * (1 - discount_rate) * (1 + tax_rate) = total_paid ∧
    regular_price = 15 := by
  sorry

end regular_price_is_15_l525_52599


namespace rebecca_egg_marble_difference_l525_52517

/-- Given that Rebecca has 20 eggs and 6 marbles, prove that she has 14 more eggs than marbles. -/
theorem rebecca_egg_marble_difference :
  let eggs : ℕ := 20
  let marbles : ℕ := 6
  eggs - marbles = 14 := by sorry

end rebecca_egg_marble_difference_l525_52517


namespace tom_rental_hours_l525_52578

/-- Represents the rental fees and total amount paid --/
structure RentalInfo where
  baseFee : ℕ
  bikeHourlyFee : ℕ
  helmetFee : ℕ
  lockHourlyFee : ℕ
  totalPaid : ℕ

/-- Calculates the number of hours rented based on the rental information --/
def calculateHoursRented (info : RentalInfo) : ℕ :=
  ((info.totalPaid - info.baseFee - info.helmetFee) / (info.bikeHourlyFee + info.lockHourlyFee))

/-- Theorem stating that Tom rented the bike and accessories for 8 hours --/
theorem tom_rental_hours (info : RentalInfo) 
    (h1 : info.baseFee = 17)
    (h2 : info.bikeHourlyFee = 7)
    (h3 : info.helmetFee = 5)
    (h4 : info.lockHourlyFee = 2)
    (h5 : info.totalPaid = 95) : 
  calculateHoursRented info = 8 := by
  sorry

end tom_rental_hours_l525_52578


namespace rectangular_plot_breadth_l525_52501

/-- 
Given a rectangular plot where:
- The area is 21 times its breadth
- The difference between the length and breadth is 10 metres
This theorem proves that the breadth of the plot is 11 metres.
-/
theorem rectangular_plot_breadth (length width : ℝ) 
  (h1 : length * width = 21 * width) 
  (h2 : length - width = 10) : 
  width = 11 := by sorry

end rectangular_plot_breadth_l525_52501


namespace sphere_volume_surface_area_ratio_l525_52505

theorem sphere_volume_surface_area_ratio 
  (r₁ r₂ : ℝ) 
  (h_positive₁ : r₁ > 0) 
  (h_positive₂ : r₂ > 0) 
  (h_volume_ratio : (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27) : 
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 := by
  sorry

#check sphere_volume_surface_area_ratio

end sphere_volume_surface_area_ratio_l525_52505


namespace cos_315_degrees_l525_52544

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_315_degrees_l525_52544


namespace product_xyz_equals_two_l525_52516

theorem product_xyz_equals_two
  (x y z : ℝ)
  (h1 : x + 1 / y = 2)
  (h2 : y + 1 / z = 2)
  (h3 : x + 1 / z = 3) :
  x * y * z = 2 := by
  sorry

end product_xyz_equals_two_l525_52516


namespace calculation_result_l525_52500

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The result of the calculation in base 10 --/
def result : Rat :=
  (toBase10 [3, 1, 0, 2] 5 : Rat) / (toBase10 [1, 1] 3) -
  (toBase10 [4, 2, 1, 3] 6 : Rat) +
  (toBase10 [1, 2, 3, 4] 7 : Rat)

theorem calculation_result : result = 898.5 := by sorry

end calculation_result_l525_52500


namespace a_neg_two_sufficient_not_necessary_l525_52569

/-- The line l₁ with equation x + ay - 2 = 0 -/
def l₁ (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + a * p.2 - 2 = 0}

/-- The line l₂ with equation (a+1)x - ay + 1 = 0 -/
def l₂ (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (a + 1) * p.1 - a * p.2 + 1 = 0}

/-- Two lines are parallel if they have the same slope -/
def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l₁ ↔ (x, k * y) ∈ l₂

/-- Theorem stating that a = -2 is a sufficient but not necessary condition for l₁ ∥ l₂ -/
theorem a_neg_two_sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ -2 ∧ parallel (l₁ a) (l₂ a)) ∧
  (parallel (l₁ (-2)) (l₂ (-2))) := by
  sorry

end a_neg_two_sufficient_not_necessary_l525_52569


namespace append_two_to_three_digit_number_l525_52502

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  is_valid : hundreds < 10 ∧ tens < 10 ∧ units < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNum (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Appends a digit to a number -/
def appendDigit (n : ℕ) (d : ℕ) : ℕ :=
  10 * n + d

theorem append_two_to_three_digit_number (n : ThreeDigitNumber) :
  appendDigit (ThreeDigitNumber.toNum n) 2 =
  1000 * n.hundreds + 100 * n.tens + 10 * n.units + 2 := by
  sorry

end append_two_to_three_digit_number_l525_52502


namespace principal_is_250_l525_52581

/-- Proves that the principal is 250 given the conditions of the problem -/
theorem principal_is_250 (P : ℝ) (I : ℝ) : 
  I = P * 0.04 * 8 →  -- Simple interest formula for 4% per annum over 8 years
  I = P - 170 →       -- Interest is 170 less than the principal
  P = 250 := by
sorry

end principal_is_250_l525_52581


namespace original_flock_size_l525_52572

/-- Represents a flock of sheep --/
structure Flock where
  rams : ℕ
  ewes : ℕ

/-- The original flock of sheep --/
def original_flock : Flock := sorry

/-- The flock after one ram runs away --/
def flock_minus_ram : Flock := 
  { rams := original_flock.rams - 1, ewes := original_flock.ewes }

/-- The flock after the ram returns and one ewe runs away --/
def flock_minus_ewe : Flock := 
  { rams := original_flock.rams, ewes := original_flock.ewes - 1 }

/-- The theorem to be proved --/
theorem original_flock_size : 
  (flock_minus_ram.rams : ℚ) / flock_minus_ram.ewes = 7 / 5 ∧
  (flock_minus_ewe.rams : ℚ) / flock_minus_ewe.ewes = 5 / 3 →
  original_flock.rams + original_flock.ewes = 25 := by
  sorry


end original_flock_size_l525_52572


namespace unique_reverse_half_ceiling_l525_52522

/-- Function that reverses the digits of an integer -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Ceiling function -/
def ceil (x : ℚ) : ℕ := sorry

theorem unique_reverse_half_ceiling :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 10000 ∧ reverse_digits n = ceil (n / 2) ∧ n = 7993 := by sorry

end unique_reverse_half_ceiling_l525_52522


namespace square_side_length_l525_52575

-- Define the right triangle PQR
def triangle_PQR (PQ PR : ℝ) : Prop := PQ = 5 ∧ PR = 12

-- Define the square on the hypotenuse
def square_on_hypotenuse (s : ℝ) (PQ PR : ℝ) : Prop :=
  ∃ (x : ℝ), 
    s / (PQ^2 + PR^2).sqrt = x / PR ∧
    s / (PR - PQ * PR / (PQ^2 + PR^2).sqrt) = (PR - x) / PR

-- Theorem statement
theorem square_side_length (PQ PR s : ℝ) : 
  triangle_PQR PQ PR →
  square_on_hypotenuse s PQ PR →
  s = 96.205 / 20.385 := by
  sorry

end square_side_length_l525_52575


namespace complex_equation_solution_l525_52503

/-- Given a and b are real numbers satisfying the equation a - bi = (1 + i)i^3,
    prove that a = 1 and b = -1. -/
theorem complex_equation_solution (a b : ℝ) :
  (a : ℂ) - b * Complex.I = (1 + Complex.I) * Complex.I^3 →
  a = 1 ∧ b = -1 := by
sorry

end complex_equation_solution_l525_52503


namespace infinite_geometric_series_first_term_l525_52559

theorem infinite_geometric_series_first_term
  (r : ℚ)
  (S : ℚ)
  (h1 : r = 1 / 4)
  (h2 : S = 20)
  (h3 : S = a / (1 - r)) :
  a = 15 := by
  sorry

end infinite_geometric_series_first_term_l525_52559


namespace ceiling_sqrt_sum_l525_52514

theorem ceiling_sqrt_sum : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 27⌉ * 2 + ⌈Real.sqrt 243⌉ = 30 := by
  sorry

end ceiling_sqrt_sum_l525_52514


namespace quadrilateral_area_is_25_l525_52583

/-- The area of the quadrilateral formed by the lines y=8, y=x+3, y=-x+3, and x=5 -/
def quadrilateralArea : ℝ := 25

/-- Line y = 8 -/
def line1 (x : ℝ) : ℝ := 8

/-- Line y = x + 3 -/
def line2 (x : ℝ) : ℝ := x + 3

/-- Line y = -x + 3 -/
def line3 (x : ℝ) : ℝ := -x + 3

/-- Line x = 5 -/
def line4 : ℝ := 5

theorem quadrilateral_area_is_25 : quadrilateralArea = 25 := by
  sorry

end quadrilateral_area_is_25_l525_52583


namespace bushes_for_sixty_zucchinis_l525_52551

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 12

/-- The number of containers of blueberries that can be traded for 3 zucchinis -/
def containers_per_trade : ℕ := 8

/-- The number of zucchinis received in one trade -/
def zucchinis_per_trade : ℕ := 3

/-- The target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- The function to calculate the number of bushes needed for a given number of zucchinis -/
def bushes_needed (zucchinis : ℕ) : ℕ :=
  (zucchinis * containers_per_trade + containers_per_bush * zucchinis_per_trade - 1) / 
  (containers_per_bush * zucchinis_per_trade)

theorem bushes_for_sixty_zucchinis :
  bushes_needed target_zucchinis = 14 := by
  sorry

end bushes_for_sixty_zucchinis_l525_52551


namespace solution_set_quadratic_inequality_l525_52510

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x * (x - 1) ≤ 0 ↔ 0 ≤ x ∧ x ≤ 1 := by sorry

end solution_set_quadratic_inequality_l525_52510


namespace y_minimized_at_b_over_2_l525_52589

variable (a b : ℝ)

def y (x : ℝ) := (x - a)^2 + (x - b)^2 + 2*(a - b)*x

theorem y_minimized_at_b_over_2 :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y a b x_min ≤ y a b x ∧ x_min = b/2 :=
sorry

end y_minimized_at_b_over_2_l525_52589


namespace polynomial_remainder_l525_52545

def polynomial (x : ℝ) : ℝ := 8*x^4 + 4*x^3 - 9*x^2 + 16*x - 28

def divisor (x : ℝ) : ℝ := 4*x - 12

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 695 := by
  sorry

end polynomial_remainder_l525_52545


namespace sqrt_two_cos_thirty_degrees_l525_52587

theorem sqrt_two_cos_thirty_degrees : 
  Real.sqrt 2 * Real.cos (30 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end sqrt_two_cos_thirty_degrees_l525_52587


namespace equilateral_triangle_side_length_l525_52541

/-- An equilateral triangle with a point inside it -/
structure EquilateralTriangleWithPoint where
  -- The side length of the equilateral triangle
  side_length : ℝ
  -- The perpendicular distances from the point to each side
  dist_to_side1 : ℝ
  dist_to_side2 : ℝ
  dist_to_side3 : ℝ
  -- Ensure the triangle is equilateral and the point is inside
  is_equilateral : side_length > 0
  point_inside : dist_to_side1 > 0 ∧ dist_to_side2 > 0 ∧ dist_to_side3 > 0

/-- The theorem to be proved -/
theorem equilateral_triangle_side_length 
  (triangle : EquilateralTriangleWithPoint) 
  (h1 : triangle.dist_to_side1 = 2) 
  (h2 : triangle.dist_to_side2 = 3) 
  (h3 : triangle.dist_to_side3 = 4) : 
  triangle.side_length = 6 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_side_length_l525_52541


namespace min_value_rational_function_l525_52585

theorem min_value_rational_function (x : ℤ) (h : x > 10) :
  (4 * x^2) / (x - 10) ≥ 160 ∧
  ((4 * x^2) / (x - 10) = 160 ↔ x = 20) :=
by sorry

end min_value_rational_function_l525_52585


namespace car_waiting_time_l525_52538

/-- Proves that a car waiting for a cyclist to catch up after 18 minutes must have initially waited 4.5 minutes -/
theorem car_waiting_time 
  (cyclist_speed : ℝ) 
  (car_speed : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : cyclist_speed = 15) 
  (h2 : car_speed = 60) 
  (h3 : catch_up_time = 18 / 60) : 
  let relative_speed := car_speed - cyclist_speed
  let distance := cyclist_speed * catch_up_time
  let initial_wait_time := distance / car_speed
  initial_wait_time * 60 = 4.5 := by sorry

end car_waiting_time_l525_52538


namespace f_increasing_f_sum_positive_l525_52558

-- Define the function f(x) = x + x^3
def f (x : ℝ) : ℝ := x + x^3

-- Theorem 1: f is an increasing function
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

-- Theorem 2: For any a, b ∈ ℝ where a + b > 0, f(a) + f(b) > 0
theorem f_sum_positive (a b : ℝ) (h : a + b > 0) : f a + f b > 0 := by sorry

end f_increasing_f_sum_positive_l525_52558


namespace gcd_324_243_l525_52513

theorem gcd_324_243 : Nat.gcd 324 243 = 81 := by
  sorry

end gcd_324_243_l525_52513


namespace isabella_babysitting_weeks_l525_52543

/-- Calculates the number of weeks Isabella has been babysitting -/
def weeks_babysitting (hourly_rate : ℚ) (hours_per_day : ℚ) (days_per_week : ℚ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (hourly_rate * hours_per_day * days_per_week)

/-- Proves that Isabella has been babysitting for 7 weeks -/
theorem isabella_babysitting_weeks :
  weeks_babysitting 5 5 6 1050 = 7 := by
  sorry

end isabella_babysitting_weeks_l525_52543


namespace product_properties_l525_52550

-- Define the range of two-digit numbers
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define the range of three-digit numbers
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Define the number of digits in a natural number
def NumDigits (n : ℕ) : ℕ := (Nat.log 10 n).succ

-- Define approximate equality
def ApproxEqual (x y : ℕ) (ε : ℕ) : Prop := (x : ℤ) - (y : ℤ) ≤ ε ∧ (y : ℤ) - (x : ℤ) ≤ ε

theorem product_properties :
  (NumDigits (52 * 403) = 5) ∧
  (ApproxEqual (52 * 403) 20000 1000) ∧
  (∀ a b, ThreeDigitNumber a → TwoDigitNumber b →
    (NumDigits (a * b) = 4 ∨ NumDigits (a * b) = 5)) :=
by sorry

end product_properties_l525_52550


namespace triangle_angle_calculation_l525_52531

theorem triangle_angle_calculation (a c : ℝ) (C : ℝ) (hA : a = 1) (hC : c = Real.sqrt 3) (hAngle : C = 2 * Real.pi / 3) :
  ∃ (A : ℝ), A = Real.pi / 6 ∧ 0 < A ∧ A < Real.pi ∧ 
  Real.sin A = a * Real.sin C / c ∧
  A + C < Real.pi :=
sorry

end triangle_angle_calculation_l525_52531


namespace existence_of_equal_differences_l525_52511

theorem existence_of_equal_differences (n : ℕ) (a : Fin (2 * n) → ℕ)
  (h_n : n ≥ 3)
  (h_a : ∀ i j : Fin (2 * n), i < j → a i < a j)
  (h_bounds : ∀ i : Fin (2 * n), 1 ≤ a i ∧ a i ≤ n^2) :
  ∃ i₁ i₂ i₃ i₄ i₅ i₆ : Fin (2 * n),
    i₁ < i₂ ∧ i₂ ≤ i₃ ∧ i₃ < i₄ ∧ i₄ ≤ i₅ ∧ i₅ < i₆ ∧
    a i₂ - a i₁ = a i₄ - a i₃ ∧ a i₄ - a i₃ = a i₆ - a i₅ :=
by sorry

end existence_of_equal_differences_l525_52511


namespace gcd_lcm_sum_75_6300_l525_52507

theorem gcd_lcm_sum_75_6300 : Nat.gcd 75 6300 + Nat.lcm 75 6300 = 6375 := by
  sorry

end gcd_lcm_sum_75_6300_l525_52507


namespace garden_area_l525_52539

/-- Represents a rectangular garden with given properties -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  length_walk : length * 20 = 1000
  perimeter_walk : (length + width) * 2 * 8 = 1000

/-- The area of a rectangular garden with the given properties is 625 square meters -/
theorem garden_area (g : RectangularGarden) : g.length * g.width = 625 := by
  sorry

end garden_area_l525_52539


namespace sin_cos_difference_45_15_l525_52530

theorem sin_cos_difference_45_15 :
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) -
  Real.cos (45 * π / 180) * Real.sin (15 * π / 180) = 1/2 := by
  sorry

end sin_cos_difference_45_15_l525_52530


namespace arithmetic_sequence_sum_specific_l525_52596

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_specific :
  let a₁ : ℤ := -3
  let d : ℤ := 6
  let n : ℕ := 8
  let aₙ : ℤ := a₁ + (n - 1) * d
  aₙ = 39 →
  arithmetic_sequence_sum a₁ d n = 144 := by
sorry

end arithmetic_sequence_sum_specific_l525_52596


namespace forty_six_in_sequence_l525_52577

def laila_sequence (n : ℕ) : ℕ :=
  4 + 7 * (n - 1)

theorem forty_six_in_sequence : ∃ n : ℕ, laila_sequence n = 46 := by
  sorry

end forty_six_in_sequence_l525_52577


namespace apartment_room_sizes_l525_52598

/-- The apartment shared by Jenny, Martha, and Sam has three rooms with a total area of 800 square feet. Jenny's room is 100 square feet larger than Martha's, and Sam's room is 50 square feet smaller than Martha's. This theorem proves that Jenny's and Sam's rooms combined have an area of 550 square feet. -/
theorem apartment_room_sizes (total_area : ℝ) (martha_size : ℝ) 
  (h1 : total_area = 800)
  (h2 : martha_size + (martha_size + 100) + (martha_size - 50) = total_area) :
  (martha_size + 100) + (martha_size - 50) = 550 := by
  sorry

end apartment_room_sizes_l525_52598


namespace exists_initial_points_for_82_l525_52593

/-- The function that calculates the number of points after one application of the procedure -/
def points_after_one_procedure (n : ℕ) : ℕ := 3 * n - 2

/-- The function that calculates the number of points after two applications of the procedure -/
def points_after_two_procedures (n : ℕ) : ℕ := 9 * n - 8

/-- Theorem stating that there exists an initial number of points that results in 82 points after two procedures -/
theorem exists_initial_points_for_82 : ∃ n : ℕ, points_after_two_procedures n = 82 := by
  sorry

end exists_initial_points_for_82_l525_52593


namespace john_climbs_70_feet_l525_52515

/-- Calculates the total height climbed by John given the number of flights, height per flight, and additional ladder length. -/
def totalHeightClimbed (numFlights : ℕ) (flightHeight : ℕ) (additionalLadderLength : ℕ) : ℕ :=
  let stairsHeight := numFlights * flightHeight
  let ropeHeight := stairsHeight / 2
  let ladderHeight := ropeHeight + additionalLadderLength
  stairsHeight + ropeHeight + ladderHeight

/-- Theorem stating that under the given conditions, John climbs a total of 70 feet. -/
theorem john_climbs_70_feet :
  totalHeightClimbed 3 10 10 = 70 := by
  sorry

end john_climbs_70_feet_l525_52515


namespace equation_solution_l525_52534

theorem equation_solution :
  let f (x : ℝ) := (7 * x + 3) / (3 * x^2 + 7 * x - 6)
  let g (x : ℝ) := (3 * x) / (3 * x - 2)
  ∀ x : ℝ, f x = g x ↔ x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end equation_solution_l525_52534


namespace abc_sum_problem_l525_52533

theorem abc_sum_problem (a b c d : ℝ) 
  (eq1 : a + b + c = 6)
  (eq2 : a + b + d = 2)
  (eq3 : a + c + d = 3)
  (eq4 : b + c + d = -3) :
  a * b + c * d = -37 / 9 := by
sorry

end abc_sum_problem_l525_52533


namespace flower_bed_fraction_l525_52588

/-- Represents a rectangular yard with two congruent isosceles right triangular flower beds -/
structure FlowerYard where
  length : ℝ
  width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  (short_side_positive : trapezoid_short_side > 0)
  (long_side_positive : trapezoid_long_side > 0)
  (short_less_than_long : trapezoid_short_side < trapezoid_long_side)
  (width_eq : width = (trapezoid_long_side - trapezoid_short_side) / 2)
  (length_eq : length = trapezoid_long_side)

/-- The fraction of the yard occupied by the flower beds is 1/5 -/
theorem flower_bed_fraction (yard : FlowerYard) (h1 : yard.trapezoid_short_side = 15) 
    (h2 : yard.trapezoid_long_side = 25) : 
  (2 * yard.width ^ 2) / (yard.length * yard.width) = 1 / 5 := by
  sorry

end flower_bed_fraction_l525_52588


namespace game_score_theorem_l525_52580

theorem game_score_theorem (a b : ℕ) (h1 : 0 < b) (h2 : b < a) (h3 : a < 1986)
  (h4 : ∀ x : ℕ, x ≥ 1986 → ∃ (m n : ℕ), x = m * a + n * b)
  (h5 : ¬∃ (m n : ℕ), 1985 = m * a + n * b)
  (h6 : ¬∃ (m n : ℕ), 663 = m * a + n * b) :
  a = 332 ∧ b = 7 := by
sorry

end game_score_theorem_l525_52580


namespace password_count_l525_52563

def password_length : ℕ := 4
def available_digits : ℕ := 9  -- digits 0-6, 8-9

def total_passwords : ℕ := available_digits ^ password_length

def passwords_with_distinct_digits : ℕ := 
  (Nat.factorial available_digits) / (Nat.factorial (available_digits - password_length))

theorem password_count : 
  total_passwords - passwords_with_distinct_digits = 3537 := by
  sorry

end password_count_l525_52563


namespace su_buqing_star_distance_l525_52591

theorem su_buqing_star_distance (d : ℝ) : d = 218000000 → d = 2.18 * (10 ^ 8) := by
  sorry

end su_buqing_star_distance_l525_52591


namespace service_cost_is_correct_l525_52532

/-- Represents the service cost per vehicle at a fuel station. -/
def service_cost_per_vehicle : ℝ := 2.30

/-- Represents the cost of fuel per liter. -/
def fuel_cost_per_liter : ℝ := 0.70

/-- Represents the number of mini-vans. -/
def num_mini_vans : ℕ := 4

/-- Represents the number of trucks. -/
def num_trucks : ℕ := 2

/-- Represents the total cost for all vehicles. -/
def total_cost : ℝ := 396

/-- Represents the capacity of a mini-van's fuel tank in liters. -/
def mini_van_tank_capacity : ℝ := 65

/-- Represents the percentage by which a truck's tank is larger than a mini-van's tank. -/
def truck_tank_percentage : ℝ := 120

/-- Theorem stating that the service cost per vehicle is $2.30 given the problem conditions. -/
theorem service_cost_is_correct :
  let truck_tank_capacity := mini_van_tank_capacity * (1 + truck_tank_percentage / 100)
  let total_fuel_volume := num_mini_vans * mini_van_tank_capacity + num_trucks * truck_tank_capacity
  let total_fuel_cost := total_fuel_volume * fuel_cost_per_liter
  let total_service_cost := total_cost - total_fuel_cost
  service_cost_per_vehicle = total_service_cost / (num_mini_vans + num_trucks) := by
  sorry


end service_cost_is_correct_l525_52532


namespace gerbil_weight_l525_52552

/-- The combined weight of two gerbils given the weights and relationships of three gerbils -/
theorem gerbil_weight (scruffy muffy puffy : ℕ) 
  (h1 : scruffy = 12)
  (h2 : muffy = scruffy - 3)
  (h3 : puffy = muffy + 5) :
  puffy + muffy = 23 := by
  sorry

end gerbil_weight_l525_52552


namespace letter_150_is_B_l525_52547

def repeating_pattern : ℕ → Char
  | n => match n % 4 with
    | 0 => 'D'
    | 1 => 'A'
    | 2 => 'B'
    | _ => 'C'

theorem letter_150_is_B : repeating_pattern 150 = 'B' := by
  sorry

end letter_150_is_B_l525_52547


namespace product_remainder_l525_52564

theorem product_remainder (x y : ℤ) 
  (hx : x % 792 = 62) 
  (hy : y % 528 = 82) : 
  (x * y) % 66 = 24 := by
  sorry

end product_remainder_l525_52564


namespace bus_journey_max_time_l525_52506

/-- Represents the transportation options available to Jenny --/
inductive TransportOption
  | Bus
  | Walk
  | Bike
  | Carpool
  | Train

/-- Calculates the total time for a given transportation option --/
def total_time (option : TransportOption) : ℝ :=
  match option with
  | .Bus => 30 + 15  -- Bus time + walking time
  | .Walk => 30
  | .Bike => 20
  | .Carpool => 25
  | .Train => 45

/-- Jenny's walking speed in miles per minute --/
def walking_speed : ℝ := 0.05

/-- The maximum time allowed for any transportation option --/
def max_allowed_time : ℝ := 45

theorem bus_journey_max_time :
  ∀ (option : TransportOption),
    total_time TransportOption.Bus ≤ max_allowed_time ∧
    total_time TransportOption.Bus = total_time option →
    30 = max_allowed_time - (0.75 / walking_speed) := by
  sorry

end bus_journey_max_time_l525_52506


namespace max_elephants_is_1036_l525_52553

/-- The number of union members --/
def union_members : ℕ := 28

/-- The number of non-union members --/
def non_union_members : ℕ := 37

/-- The function that calculates the total number of elephants given the number
    of elephants per union member and per non-union member --/
def total_elephants (elephants_per_union : ℕ) (elephants_per_non_union : ℕ) : ℕ :=
  union_members * elephants_per_union + non_union_members * elephants_per_non_union

/-- The theorem stating that 1036 is the maximum number of elephants that can be distributed --/
theorem max_elephants_is_1036 :
  ∃ (eu en : ℕ), 
    eu ≠ en ∧ 
    eu > 0 ∧ 
    en > 0 ∧
    total_elephants eu en = 1036 ∧
    (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → total_elephants x y ≤ 1036) :=
sorry

end max_elephants_is_1036_l525_52553


namespace algebraic_simplification_l525_52520

theorem algebraic_simplification (a b : ℝ) : -3*a*(2*a - 4*b + 2) + 6*a = -6*a^2 + 12*a*b := by
  sorry

end algebraic_simplification_l525_52520


namespace ned_bomb_diffusion_l525_52521

/-- Ned's bomb diffusion problem -/
theorem ned_bomb_diffusion (total_flights : ℕ) (time_per_flight : ℕ) (bomb_timer : ℕ) (time_spent : ℕ)
  (h1 : total_flights = 20)
  (h2 : time_per_flight = 11)
  (h3 : bomb_timer = 72)
  (h4 : time_spent = 165) :
  bomb_timer - (total_flights - time_spent / time_per_flight) * time_per_flight = 17 := by
  sorry

#check ned_bomb_diffusion

end ned_bomb_diffusion_l525_52521


namespace planes_perpendicular_l525_52528

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : parallel m n) 
  (h4 : parallel_plane m α) 
  (h5 : perpendicular n β) : 
  perpendicular_plane α β :=
sorry

end planes_perpendicular_l525_52528


namespace avg_cost_rounded_to_13_l525_52508

/- Define the number of pencils -/
def num_pencils : ℕ := 200

/- Define the cost of pencils in cents -/
def pencil_cost : ℕ := 1990

/- Define the shipping cost in cents -/
def shipping_cost : ℕ := 695

/- Define the function to calculate the average cost per pencil in cents -/
def avg_cost_per_pencil : ℚ :=
  (pencil_cost + shipping_cost : ℚ) / num_pencils

/- Define the function to round to the nearest whole number -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/- Theorem statement -/
theorem avg_cost_rounded_to_13 :
  round_to_nearest avg_cost_per_pencil = 13 := by
  sorry


end avg_cost_rounded_to_13_l525_52508


namespace boatman_distance_along_current_l525_52548

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  stationary : ℝ
  against_current : ℝ
  current : ℝ
  along_current : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (speed time : ℝ) : ℝ := speed * time

/-- Theorem: The boatman travels 1 km along the current -/
theorem boatman_distance_along_current 
  (speed : BoatSpeed)
  (h1 : distance speed.against_current 4 = 4) -- 4 km against current in 4 hours
  (h2 : distance speed.stationary 3 = 6)      -- 6 km in stationary water in 3 hours
  (h3 : speed.current = speed.stationary - speed.against_current)
  (h4 : speed.along_current = speed.stationary + speed.current)
  : distance speed.along_current (1/3) = 1 := by
  sorry

end boatman_distance_along_current_l525_52548


namespace closest_integer_to_cube_root_250_l525_52567

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n - (250 : ℝ)^(1/3)| ≥ |(6 : ℝ) - (250 : ℝ)^(1/3)| :=
by sorry

end closest_integer_to_cube_root_250_l525_52567


namespace fraction_of_boys_l525_52554

theorem fraction_of_boys (total_students : ℕ) (girls_no_pets : ℕ) 
  (dog_owners_percent : ℚ) (cat_owners_percent : ℚ) :
  total_students = 30 →
  girls_no_pets = 8 →
  dog_owners_percent = 40 / 100 →
  cat_owners_percent = 20 / 100 →
  (17 : ℚ) / 30 = (total_students - (girls_no_pets / ((1 : ℚ) - dog_owners_percent - cat_owners_percent))) / total_students :=
by sorry

end fraction_of_boys_l525_52554


namespace part_one_part_two_l525_52561

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - (a + 1/a)*x + 1 < 0

def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Part 1
theorem part_one (x : ℝ) (h1 : p x 2) (h2 : q x) : 1 ≤ x ∧ x < 2 :=
sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, q x → p x a) (h_not_sufficient : ¬(∀ x, p x a → q x)) : 3 < a :=
sorry

end part_one_part_two_l525_52561


namespace fraction_zero_implies_x_two_l525_52565

theorem fraction_zero_implies_x_two (x : ℝ) : 
  (x^2 - 4) / (x + 2) = 0 → x = 2 := by
  sorry

end fraction_zero_implies_x_two_l525_52565


namespace initial_number_proof_l525_52560

theorem initial_number_proof (x : ℝ) : (x - 1/4) / (1/2) = 4.5 → x = 2.5 := by
  sorry

end initial_number_proof_l525_52560


namespace paint_remaining_is_three_eighths_l525_52509

/-- The fraction of paint remaining after two days of use -/
def paint_remaining (initial_amount : ℚ) (first_day_use : ℚ) (second_day_use : ℚ) : ℚ :=
  initial_amount - (first_day_use * initial_amount) - (second_day_use * (initial_amount - first_day_use * initial_amount))

/-- Theorem stating that the fraction of paint remaining after two days is 3/8 -/
theorem paint_remaining_is_three_eighths :
  paint_remaining 1 (1/4) (1/2) = 3/8 := by
  sorry

end paint_remaining_is_three_eighths_l525_52509


namespace geometric_sequence_property_l525_52537

def is_geometric_sequence (a : Fin 4 → ℝ) : Prop :=
  ∃ q : ℝ, ∀ i : Fin 3, a (i + 1) = a i * q

theorem geometric_sequence_property
  (a : Fin 4 → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 = Real.log (a 0 + a 1 + a 2))
  (h_a1 : a 0 > 1) :
  a 0 > a 2 ∧ a 1 < a 3 :=
sorry

end geometric_sequence_property_l525_52537


namespace largest_six_digit_with_factorial_product_l525_52540

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).foldl (· * ·) 1

theorem largest_six_digit_with_factorial_product :
  ∃ (n : ℕ), 
    100000 ≤ n ∧ 
    n ≤ 999999 ∧ 
    digit_product n = factorial 8 ∧
    ∀ (m : ℕ), 100000 ≤ m ∧ m ≤ 999999 ∧ digit_product m = factorial 8 → m ≤ n :=
by
  use 987542
  sorry

end largest_six_digit_with_factorial_product_l525_52540


namespace prime_squared_minus_one_divisible_by_24_l525_52527

theorem prime_squared_minus_one_divisible_by_24 (n : ℕ) 
  (h_prime : Nat.Prime n) (h_not_two : n ≠ 2) (h_not_three : n ≠ 3) :
  24 ∣ (n^2 - 1) := by
  sorry

end prime_squared_minus_one_divisible_by_24_l525_52527


namespace missing_number_is_1745_l525_52594

def known_numbers : List ℕ := [744, 747, 748, 749, 752, 752, 753, 755, 755]

theorem missing_number_is_1745 :
  let total_count : ℕ := 10
  let average : ℕ := 750
  let sum_known : ℕ := known_numbers.sum
  let missing_number : ℕ := total_count * average - sum_known
  missing_number = 1745 := by sorry

end missing_number_is_1745_l525_52594


namespace max_squares_covered_l525_52584

/-- Represents a square card with a given side length -/
structure Card where
  side : ℝ
  side_positive : side > 0

/-- Represents a square on a checkerboard -/
structure CheckerboardSquare where
  side : ℝ
  side_positive : side > 0

/-- Calculates the number of checkerboard squares covered by a card -/
def squares_covered (card : Card) (square : CheckerboardSquare) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a 1-inch checkerboard -/
theorem max_squares_covered :
  let card := Card.mk 2 (by norm_num)
  let square := CheckerboardSquare.mk 1 (by norm_num)
  ∀ n : ℕ, squares_covered card square ≤ n → n ≤ 8 :=
sorry

end max_squares_covered_l525_52584


namespace least_marbles_count_l525_52566

theorem least_marbles_count (n : ℕ) : n ≥ 402 →
  (n % 7 = 3 ∧ n % 4 = 2 ∧ n % 6 = 1) →
  n = 402 :=
by sorry

end least_marbles_count_l525_52566


namespace die_probability_l525_52512

theorem die_probability (total_faces : ℕ) (red_faces : ℕ) (yellow_faces : ℕ) (blue_faces : ℕ)
  (h1 : total_faces = 11)
  (h2 : red_faces = 5)
  (h3 : yellow_faces = 4)
  (h4 : blue_faces = 2)
  (h5 : total_faces = red_faces + yellow_faces + blue_faces) :
  (yellow_faces : ℚ) / total_faces * (blue_faces : ℚ) / total_faces = 8 / 121 := by
  sorry

end die_probability_l525_52512


namespace dani_initial_pants_l525_52570

/-- Represents the number of pants Dani receives each year as a reward -/
def yearly_reward : ℕ := 4 * 2

/-- Represents the number of years -/
def years : ℕ := 5

/-- Represents the total number of pants Dani will have after 5 years -/
def total_pants : ℕ := 90

/-- Calculates the number of pants Dani initially had -/
def initial_pants : ℕ := total_pants - (yearly_reward * years)

theorem dani_initial_pants :
  initial_pants = 50 := by sorry

end dani_initial_pants_l525_52570


namespace shoe_tying_time_difference_l525_52542

theorem shoe_tying_time_difference (jack_shoe_time toddler_count total_time : ℕ) :
  jack_shoe_time = 4 →
  toddler_count = 2 →
  total_time = 18 →
  (total_time - jack_shoe_time) / toddler_count - jack_shoe_time = 3 := by
  sorry

end shoe_tying_time_difference_l525_52542


namespace chicken_egg_production_l525_52526

theorem chicken_egg_production 
  (num_chickens : ℕ) 
  (price_per_dozen : ℚ) 
  (total_revenue : ℚ) 
  (num_weeks : ℕ) 
  (h1 : num_chickens = 46)
  (h2 : price_per_dozen = 3)
  (h3 : total_revenue = 552)
  (h4 : num_weeks = 8) :
  (total_revenue / (price_per_dozen / 12) / num_weeks / num_chickens : ℚ) = 6 := by
  sorry

end chicken_egg_production_l525_52526


namespace chessboard_constant_l525_52592

/-- A function representing the numbers on an infinite chessboard. -/
def ChessboardFunction := ℤ × ℤ → ℝ

/-- The property that each number is the arithmetic mean of its four neighbors. -/
def IsMeanValue (f : ChessboardFunction) : Prop :=
  ∀ m n : ℤ, f (m, n) = (f (m+1, n) + f (m-1, n) + f (m, n+1) + f (m, n-1)) / 4

/-- The property that all values of the function are nonnegative. -/
def IsNonnegative (f : ChessboardFunction) : Prop :=
  ∀ m n : ℤ, 0 ≤ f (m, n)

/-- Theorem stating that a nonnegative function satisfying the mean value property is constant. -/
theorem chessboard_constant (f : ChessboardFunction) 
  (h_mean : IsMeanValue f) (h_nonneg : IsNonnegative f) : 
  ∃ c : ℝ, ∀ m n : ℤ, f (m, n) = c :=
sorry

end chessboard_constant_l525_52592


namespace length_BC_is_44_div_3_l525_52595

/-- Two externally tangent circles with a common external tangent line -/
structure TangentCircles where
  /-- Center of the first circle -/
  A : ℝ × ℝ
  /-- Center of the second circle -/
  B : ℝ × ℝ
  /-- Radius of the first circle -/
  r₁ : ℝ
  /-- Radius of the second circle -/
  r₂ : ℝ
  /-- Point where the external tangent line intersects ray AB -/
  C : ℝ × ℝ
  /-- The circles are externally tangent -/
  externally_tangent : dist A B = r₁ + r₂
  /-- The line through C is externally tangent to both circles -/
  is_external_tangent : ∃ (D E : ℝ × ℝ), 
    dist A D = r₁ ∧ dist B E = r₂ ∧ 
    (C.1 - D.1) * (A.1 - D.1) + (C.2 - D.2) * (A.2 - D.2) = 0 ∧
    (C.1 - E.1) * (B.1 - E.1) + (C.2 - E.2) * (B.2 - E.2) = 0
  /-- C lies on ray AB -/
  C_on_ray_AB : ∃ (t : ℝ), t ≥ 0 ∧ C = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

/-- The length of BC in the TangentCircles configuration -/
def length_BC (tc : TangentCircles) : ℝ :=
  dist tc.B tc.C

/-- The main theorem: length of BC is 44/3 -/
theorem length_BC_is_44_div_3 (tc : TangentCircles) (h₁ : tc.r₁ = 7) (h₂ : tc.r₂ = 4) : 
  length_BC tc = 44 / 3 := by
  sorry


end length_BC_is_44_div_3_l525_52595


namespace polynomial_division_problem_l525_52574

theorem polynomial_division_problem (a : ℤ) : 
  (∃ p : Polynomial ℤ, (X^2 - 2*X + a) * p = X^13 + 2*X + 180) ↔ a = 3 :=
sorry

end polynomial_division_problem_l525_52574


namespace competition_winner_and_probability_l525_52504

def prob_A_win_round1 : ℚ := 3/5
def prob_B_win_round1 : ℚ := 3/4
def prob_A_win_round2 : ℚ := 3/5
def prob_B_win_round2 : ℚ := 1/2

def prob_A_win_competition : ℚ := prob_A_win_round1 * prob_A_win_round2
def prob_B_win_competition : ℚ := prob_B_win_round1 * prob_B_win_round2

theorem competition_winner_and_probability :
  (prob_B_win_competition > prob_A_win_competition) ∧
  (1 - (1 - prob_A_win_competition) * (1 - prob_B_win_competition) = 3/5) := by
  sorry

end competition_winner_and_probability_l525_52504


namespace solution_set_inequality_l525_52536

theorem solution_set_inequality (x : ℝ) : 
  (2 * x) / (x + 1) ≤ 1 ↔ x ∈ Set.Ioc (-1) 1 := by
  sorry

end solution_set_inequality_l525_52536
