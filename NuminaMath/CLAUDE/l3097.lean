import Mathlib

namespace existence_of_special_set_l3097_309720

theorem existence_of_special_set (n : ℕ) (p : ℕ) (h_n : n ≥ 2) (h_p : Nat.Prime p) (h_div : p ∣ n) :
  ∃ (A : Fin n → ℕ), ∀ (i j : Fin n) (S : Finset (Fin n)), S.card = p →
    (A i * A j) ∣ (S.sum A) :=
sorry

end existence_of_special_set_l3097_309720


namespace tan_theta_value_l3097_309776

theorem tan_theta_value (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2))
  (h2 : 12 / Real.sin θ + 12 / Real.cos θ = 35) :
  Real.tan θ = 3/4 ∨ Real.tan θ = 4/3 := by
  sorry

end tan_theta_value_l3097_309776


namespace dogSchoolCount_l3097_309793

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  sit : ℕ
  stay : ℕ
  roll : ℕ
  sitStay : ℕ
  stayRoll : ℕ
  sitRoll : ℕ
  allThree : ℕ
  none : ℕ

/-- Calculates the total number of dogs in the school -/
def totalDogs (d : DogTricks) : ℕ :=
  d.allThree +
  (d.sitRoll - d.allThree) +
  (d.stayRoll - d.allThree) +
  (d.sitStay - d.allThree) +
  (d.sit - d.sitRoll - d.sitStay + d.allThree) +
  (d.stay - d.stayRoll - d.sitStay + d.allThree) +
  (d.roll - d.sitRoll - d.stayRoll + d.allThree) +
  d.none

/-- Theorem stating that the total number of dogs in the school is 84 -/
theorem dogSchoolCount (d : DogTricks)
  (h1 : d.sit = 50)
  (h2 : d.stay = 29)
  (h3 : d.roll = 34)
  (h4 : d.sitStay = 17)
  (h5 : d.stayRoll = 12)
  (h6 : d.sitRoll = 18)
  (h7 : d.allThree = 9)
  (h8 : d.none = 9) :
  totalDogs d = 84 := by
  sorry

end dogSchoolCount_l3097_309793


namespace substitution_ways_soccer_l3097_309731

/-- The number of ways a coach can make substitutions in a soccer game -/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitutes := total_players - starting_players
  let no_sub := 1
  let one_sub := starting_players * substitutes
  let two_sub := one_sub * (starting_players - 1) * (substitutes + 1)
  let three_sub := two_sub * (starting_players - 2) * (substitutes + 2)
  let four_sub := three_sub * (starting_players - 3) * (substitutes + 3)
  (no_sub + one_sub + two_sub + three_sub + four_sub) % 1000

theorem substitution_ways_soccer : 
  substitution_ways 25 14 4 = 
  (1 + 14 * 11 + 14 * 11 * 13 * 12 + 14 * 11 * 13 * 12 * 12 * 13 + 
   14 * 11 * 13 * 12 * 12 * 13 * 11 * 14) % 1000 := by
  sorry

end substitution_ways_soccer_l3097_309731


namespace parallelograms_in_divided_triangle_l3097_309742

/-- The number of parallelograms formed in a triangle with sides divided into n equal parts -/
def num_parallelograms (n : ℕ) : ℕ :=
  3 * (Nat.choose (n + 2) 4)

/-- Theorem stating the number of parallelograms in a divided triangle -/
theorem parallelograms_in_divided_triangle (n : ℕ) :
  num_parallelograms n = 3 * (Nat.choose (n + 2) 4) :=
by sorry

end parallelograms_in_divided_triangle_l3097_309742


namespace min_perimeter_rectangle_min_perimeter_achieved_l3097_309779

theorem min_perimeter_rectangle (length width : ℝ) : 
  length > 0 → width > 0 → length * width = 64 → 
  2 * (length + width) ≥ 32 := by
  sorry

theorem min_perimeter_achieved (length width : ℝ) :
  length > 0 → width > 0 → length * width = 64 →
  2 * (length + width) = 32 ↔ length = 8 ∧ width = 8 := by
  sorry

end min_perimeter_rectangle_min_perimeter_achieved_l3097_309779


namespace range_of_x_when_m_is_4_range_of_m_l3097_309769

/-- Definition of proposition p -/
def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

/-- Definition of proposition q -/
def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

/-- Theorem for part (1) -/
theorem range_of_x_when_m_is_4 (x : ℝ) :
  (∃ m : ℝ, m > 0 ∧ m = 4 ∧ p x ∧ q x m) → 4 < x ∧ x < 5 := by sorry

/-- Theorem for part (2) -/
theorem range_of_m (m : ℝ) :
  (m > 0 ∧ (∀ x : ℝ, ¬(q x m) → ¬(p x)) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m)) →
  (5/3 ≤ m ∧ m ≤ 2) := by sorry

end range_of_x_when_m_is_4_range_of_m_l3097_309769


namespace last_draw_same_color_prob_l3097_309738

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 2

/-- Represents the number of colors of marbles -/
def num_colors : ℕ := 3

/-- Represents the total number of marbles -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- Represents the number of people drawing marbles -/
def num_people : ℕ := 3

/-- Represents the number of marbles each person draws -/
def marbles_per_draw : ℕ := 2

/-- Calculates the probability of the last person drawing two marbles of the same color -/
def prob_last_draw_same_color : ℚ :=
  (num_colors * (Nat.choose (total_marbles - 2 * marbles_per_draw) marbles_per_draw)) /
  (Nat.choose total_marbles marbles_per_draw * 
   Nat.choose (total_marbles - marbles_per_draw) marbles_per_draw)

theorem last_draw_same_color_prob :
  prob_last_draw_same_color = 1 / 5 := by sorry

end last_draw_same_color_prob_l3097_309738


namespace perpendicular_lines_sum_l3097_309756

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y - 2 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := 2 * x - 5 * y + b = 0

-- Define perpendicularity of two lines
def perpendicular (a b : ℝ) : Prop := a * 2 + 4 * (-5) = 0

-- Define the foot of the perpendicular
def foot_of_perpendicular (a b c : ℝ) : Prop := l₁ a 1 c ∧ l₂ b 1 c

-- Theorem statement
theorem perpendicular_lines_sum (a b c : ℝ) :
  perpendicular a b →
  foot_of_perpendicular a b c →
  a + b + c = -4 := by sorry

end perpendicular_lines_sum_l3097_309756


namespace retailer_profit_percentage_l3097_309791

theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_percentage : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : discount_percentage = 0.1)
  : (((retail_price * (1 - discount_percentage)) - wholesale_price) / wholesale_price) * 100 = 20 := by
  sorry

end retailer_profit_percentage_l3097_309791


namespace sum_of_x_coordinates_P_is_640_l3097_309759

-- Define the points
def Q : ℝ × ℝ := (0, 0)
def R : ℝ × ℝ := (307, 0)
def S : ℝ × ℝ := (450, 280)
def T : ℝ × ℝ := (460, 290)

-- Define the areas of the triangles
def area_PQR : ℝ := 1739
def area_PST : ℝ := 6956

-- Define the function to calculate the sum of possible x-coordinates of P
noncomputable def sum_of_x_coordinates_P : ℝ := sorry

-- Theorem statement
theorem sum_of_x_coordinates_P_is_640 :
  sum_of_x_coordinates_P = 640 := by sorry

end sum_of_x_coordinates_P_is_640_l3097_309759


namespace omega_value_for_max_sine_l3097_309732

theorem omega_value_for_max_sine (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∀ x ∈ Set.Icc 0 (π / 3), 2 * Real.sin (ω * x) ≤ Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (π / 3), 2 * Real.sin (ω * x) = Real.sqrt 2) →
  ω = 3 / 4 := by
sorry

end omega_value_for_max_sine_l3097_309732


namespace garrison_size_l3097_309717

theorem garrison_size (initial_days : ℕ) (reinforcement_size : ℕ) (days_before_reinforcement : ℕ) (remaining_days : ℕ) :
  initial_days = 62 →
  reinforcement_size = 2700 →
  days_before_reinforcement = 15 →
  remaining_days = 20 →
  ∃ (initial_men : ℕ),
    initial_men * (initial_days - days_before_reinforcement) = 
    (initial_men + reinforcement_size) * remaining_days ∧
    initial_men = 2000 :=
by
  sorry

end garrison_size_l3097_309717


namespace geometric_series_common_ratio_l3097_309773

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 16/49
  let a₃ : ℚ := 64/343
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → (7^n * a₁ = 4^n)) → r = 4/7 := by
  sorry

end geometric_series_common_ratio_l3097_309773


namespace angle_equivalence_l3097_309771

/-- Proves that 2023° is equivalent to -137° in the context of angle measurements -/
theorem angle_equivalence : ∃ (k : ℤ), 2023 = -137 + 360 * k := by sorry

end angle_equivalence_l3097_309771


namespace polygon_120_sides_diagonals_l3097_309702

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 120 sides has 7020 diagonals -/
theorem polygon_120_sides_diagonals :
  num_diagonals 120 = 7020 := by
  sorry

end polygon_120_sides_diagonals_l3097_309702


namespace least_integer_with_remainders_l3097_309775

theorem least_integer_with_remainders : ∃! n : ℕ,
  (∀ m : ℕ, m < n →
    (m % 5 ≠ 4 ∨ m % 6 ≠ 5 ∨ m % 7 ≠ 6 ∨ m % 8 ≠ 7 ∨ m % 9 ≠ 8 ∨ m % 10 ≠ 9)) ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  n = 2519 :=
sorry

end least_integer_with_remainders_l3097_309775


namespace sheep_count_l3097_309718

theorem sheep_count (total animals : ℕ) (cows goats : ℕ) 
  (h1 : total = 200)
  (h2 : cows = 40)
  (h3 : goats = 104)
  (h4 : animals = total - cows - goats) :
  animals = 56 := by
  sorry

end sheep_count_l3097_309718


namespace production_rates_l3097_309768

/-- The production rates of two workers --/
theorem production_rates (total_rate : ℝ) (a_parts b_parts : ℕ) 
  (h1 : total_rate = 35)
  (h2 : (a_parts : ℝ) / x = (b_parts : ℝ) / (total_rate - x))
  (h3 : a_parts = 90)
  (h4 : b_parts = 120) :
  ∃ (x y : ℝ), x + y = total_rate ∧ x = 15 ∧ y = 20 :=
sorry

end production_rates_l3097_309768


namespace inverse_g_one_over_120_l3097_309784

noncomputable def g (x : ℝ) : ℝ := (x^5 + 1) / 5

theorem inverse_g_one_over_120 :
  g⁻¹ (1/120) = ((-23/24) : ℝ)^(1/5) :=
by sorry

end inverse_g_one_over_120_l3097_309784


namespace weeks_passed_l3097_309735

/-- Prove that the number of weeks that have already passed is 4 --/
theorem weeks_passed
  (watch_cost : ℕ)
  (weekly_allowance : ℕ)
  (current_savings : ℕ)
  (weeks_left : ℕ)
  (h1 : watch_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : current_savings = 20)
  (h4 : weeks_left = 16)
  (h5 : current_savings + weeks_left * weekly_allowance = watch_cost) :
  current_savings / weekly_allowance = 4 := by
  sorry


end weeks_passed_l3097_309735


namespace least_N_for_prime_condition_l3097_309783

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a multiple of 12 -/
def isMultipleOf12 (n : ℕ) : Prop := sorry

/-- The theorem statement -/
theorem least_N_for_prime_condition : 
  ∃ (N : ℕ), N > 0 ∧ 
  (∀ (n : ℕ), isPrime (1 + N * 2^n) ↔ isMultipleOf12 n) ∧
  (∀ (M : ℕ), M > 0 → M < N → 
    ¬(∀ (n : ℕ), isPrime (1 + M * 2^n) ↔ isMultipleOf12 n)) ∧
  N = 556 := by
  sorry

end least_N_for_prime_condition_l3097_309783


namespace martha_guess_probability_l3097_309782

/-- Martha's guessing abilities -/
structure MarthaGuess where
  height_success : Rat
  weight_success : Rat
  child_height_success : Rat
  adult_height_success : Rat
  tight_clothes_weight_success : Rat
  loose_clothes_weight_success : Rat

/-- Represents a person Martha meets -/
inductive Person
  | Child : Bool → Person  -- Bool represents tight (true) or loose (false) clothes
  | Adult : Bool → Person

def martha : MarthaGuess :=
  { height_success := 5/6
  , weight_success := 6/8
  , child_height_success := 4/5
  , adult_height_success := 5/6
  , tight_clothes_weight_success := 3/4
  , loose_clothes_weight_success := 7/10 }

def people : List Person :=
  [Person.Child false, Person.Adult true, Person.Adult false]

/-- Calculates the probability of Martha guessing correctly for a specific person -/
def guessCorrectProb (m : MarthaGuess) (p : Person) : Rat :=
  match p with
  | Person.Child tight =>
      1 - (1 - m.child_height_success) * (1 - (if tight then m.tight_clothes_weight_success else m.loose_clothes_weight_success))
  | Person.Adult tight =>
      1 - (1 - m.adult_height_success) * (1 - (if tight then m.tight_clothes_weight_success else m.loose_clothes_weight_success))

/-- Theorem: The probability of Martha guessing correctly at least once for the given people is 7999/8000 -/
theorem martha_guess_probability :
  1 - (people.map (guessCorrectProb martha)).prod = 7999/8000 := by
  sorry


end martha_guess_probability_l3097_309782


namespace power_division_negative_x_l3097_309754

theorem power_division_negative_x (x : ℝ) : (-x)^8 / (-x)^4 = x^4 := by sorry

end power_division_negative_x_l3097_309754


namespace inverse_proportion_exists_l3097_309794

theorem inverse_proportion_exists (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁ < 0) (h2 : 0 < x₂) (h3 : y₁ > y₂) : 
  ∃ k : ℝ, k < 0 ∧ y₁ = k / x₁ ∧ y₂ = k / x₂ := by
  sorry

end inverse_proportion_exists_l3097_309794


namespace no_solution_exists_l3097_309700

theorem no_solution_exists : ¬∃ (x : ℤ), x^2 = 3*x + 75 := by
  sorry

end no_solution_exists_l3097_309700


namespace sum_of_cubes_of_roots_l3097_309725

/-- Given a polynomial x^4 + 5x^3 + 6x^2 + 5x + 1 with complex roots, 
    the sum of the cubes of its roots is -54 -/
theorem sum_of_cubes_of_roots : 
  ∀ (x₁ x₂ x₃ x₄ : ℂ), 
    (x₁^4 + 5*x₁^3 + 6*x₁^2 + 5*x₁ + 1 = 0) →
    (x₂^4 + 5*x₂^3 + 6*x₂^2 + 5*x₂ + 1 = 0) →
    (x₃^4 + 5*x₃^3 + 6*x₃^2 + 5*x₃ + 1 = 0) →
    (x₄^4 + 5*x₄^3 + 6*x₄^2 + 5*x₄ + 1 = 0) →
    x₁^3 + x₂^3 + x₃^3 + x₄^3 = -54 :=
by sorry

end sum_of_cubes_of_roots_l3097_309725


namespace weight_plate_problem_l3097_309799

theorem weight_plate_problem (num_plates : ℕ) (weight_increase : ℝ) (felt_weight : ℝ) :
  num_plates = 10 →
  weight_increase = 0.2 →
  felt_weight = 360 →
  (felt_weight / (1 + weight_increase)) / num_plates = 30 := by
  sorry

end weight_plate_problem_l3097_309799


namespace paint_fraction_first_week_l3097_309797

/-- Proves that the fraction of paint used in the first week is 1/9 -/
theorem paint_fraction_first_week (total_paint : ℝ) (paint_used : ℝ) 
  (h1 : total_paint = 360)
  (h2 : paint_used = 104)
  (h3 : ∀ f : ℝ, paint_used = f * total_paint + 1/5 * (total_paint - f * total_paint)) :
  ∃ f : ℝ, f = 1/9 ∧ paint_used = f * total_paint + 1/5 * (total_paint - f * total_paint) :=
by sorry

end paint_fraction_first_week_l3097_309797


namespace intersection_trajectory_l3097_309755

/-- The trajectory of the intersection point of two rotating rods -/
theorem intersection_trajectory (a : ℝ) (h : a ≠ 0) :
  ∃ (x y : ℝ), 
    (∃ (b b₁ : ℝ), b * b₁ = a^2 ∧ b ≠ 0 ∧ b₁ ≠ 0) →
    (y = -b / a * (x - a) ∧ y = b₁ / a * (x + a)) →
    x^2 + y^2 = a^2 ∧ -a < x ∧ x < a :=
by sorry

end intersection_trajectory_l3097_309755


namespace total_fat_served_l3097_309737

/-- The amount of fat in ounces for a single herring -/
def herring_fat : ℕ := 40

/-- The amount of fat in ounces for a single eel -/
def eel_fat : ℕ := 20

/-- The amount of fat in ounces for a single pike -/
def pike_fat : ℕ := eel_fat + 10

/-- The number of fish of each type served -/
def fish_count : ℕ := 40

/-- The total amount of fat served in ounces -/
def total_fat : ℕ := fish_count * (herring_fat + eel_fat + pike_fat)

theorem total_fat_served :
  total_fat = 3600 := by sorry

end total_fat_served_l3097_309737


namespace inverse_g_84_l3097_309706

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- State the theorem
theorem inverse_g_84 : g⁻¹ 84 = 3 := by sorry

end inverse_g_84_l3097_309706


namespace test_score_problem_l3097_309780

/-- Prove that given a test with 30 questions, where each correct answer is worth 20 points
    and each incorrect answer deducts 5 points, if all questions are answered and the total
    score is 325, then the number of correct answers is 19. -/
theorem test_score_problem (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) 
    (total_score : ℕ) (h1 : total_questions = 30) (h2 : correct_points = 20) 
    (h3 : incorrect_points = 5) (h4 : total_score = 325) : 
    ∃ (correct_answers : ℕ), 
      correct_answers * correct_points + 
      (total_questions - correct_answers) * (correct_points - incorrect_points) = 
      total_score ∧ correct_answers = 19 := by
  sorry

end test_score_problem_l3097_309780


namespace sqrt_equality_condition_l3097_309726

theorem sqrt_equality_condition (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  Real.sqrt (a - b + c) = Real.sqrt a - Real.sqrt b + Real.sqrt c ↔ a = b ∨ b = c :=
sorry

end sqrt_equality_condition_l3097_309726


namespace circle_equations_valid_l3097_309751

-- Define the points
def M : ℝ × ℝ := (-1, 1)
def N : ℝ × ℝ := (0, 2)
def Q : ℝ × ℝ := (2, 0)

-- Define the equations of the circles
def circle_C1_eq (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1/2)^2 = 5/2
def circle_C2_eq (x y : ℝ) : Prop := (x + 3/2)^2 + (y - 5/2)^2 = 5/2

-- Define the line MN
def line_MN_eq (x y : ℝ) : Prop := x - y + 2 = 0

-- Theorem statement
theorem circle_equations_valid :
  -- Circle C1 passes through M, N, and Q
  (circle_C1_eq M.1 M.2 ∧ circle_C1_eq N.1 N.2 ∧ circle_C1_eq Q.1 Q.2) ∧
  -- C2 is the reflection of C1 about line MN
  (∀ x y : ℝ, circle_C1_eq x y ↔ 
    ∃ x' y' : ℝ, circle_C2_eq x' y' ∧ 
    ((x + x')/2 - (y + y')/2 + 2 = 0) ∧
    (y' - y)/(x' - x) = -1) :=
sorry

end circle_equations_valid_l3097_309751


namespace range_of_a_l3097_309730

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - a| ≥ 5) → 
  a ∈ Set.Ici 4 ∪ Set.Iic (-6) := by
sorry

end range_of_a_l3097_309730


namespace a_6_value_l3097_309781

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem a_6_value (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 2 + a 10 = -12) (h3 : a 2 * a 10 = -8) : a 6 = -6 := by
  sorry

end a_6_value_l3097_309781


namespace playground_children_count_l3097_309746

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 44) 
  (h2 : girls = 53) : 
  boys + girls = 97 := by
sorry

end playground_children_count_l3097_309746


namespace arithmetic_sequence_property_l3097_309767

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 2 + a 5 = 18
  product_property : a 3 * a 4 = 32

/-- The theorem stating that for the given arithmetic sequence, a_n = 128 implies n = 8 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  ∃ n : ℕ, seq.a n = 128 → n = 8 := by
  sorry

end arithmetic_sequence_property_l3097_309767


namespace four_fold_f_of_two_plus_i_l3097_309703

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z ^ 2 else -(z ^ 2)

-- State the theorem
theorem four_fold_f_of_two_plus_i :
  f (f (f (f (2 + Complex.I)))) = 164833 + 354192 * Complex.I := by
  sorry

end four_fold_f_of_two_plus_i_l3097_309703


namespace power_function_coefficient_l3097_309750

theorem power_function_coefficient (m : ℝ) : 
  (∃ (y : ℝ → ℝ), ∀ x, y x = (m^2 + 2*m - 2) * x^4 ∧ ∃ (k : ℝ), ∀ x, y x = x^k) → 
  m = 1 ∨ m = -3 :=
by sorry

end power_function_coefficient_l3097_309750


namespace ratio_of_percentages_l3097_309712

theorem ratio_of_percentages (P M N R : ℝ) : 
  P > 0 ∧ P = 0.3 * R ∧ M = 0.35 * R ∧ N = 0.55 * R → M / N = 7 / 11 :=
by
  sorry

end ratio_of_percentages_l3097_309712


namespace principal_amount_l3097_309798

/-- Proves that given the conditions of the problem, the principal amount is 300 --/
theorem principal_amount (P : ℝ) : 
  (P * 4 * 8 / 100 = P - 204) → P = 300 := by
  sorry

end principal_amount_l3097_309798


namespace geometric_series_ratio_l3097_309762

/-- For an infinite geometric series with first term a and common ratio r,
    if the sum of the series starting from the fourth term is 1/27 times
    the sum of the original series, then r = 1/3. -/
theorem geometric_series_ratio (a r : ℝ) (h : |r| < 1) :
  (a * r^3 / (1 - r)) = (1 / 27) * (a / (1 - r)) →
  r = 1 / 3 := by
sorry

end geometric_series_ratio_l3097_309762


namespace inequality_solution_l3097_309789

theorem inequality_solution (x : ℤ) : 
  (3 * x - 5 ≤ 10 - 2 * x) ↔ x ∈ ({-2, -1, 0, 1, 2, 3} : Set ℤ) := by sorry

end inequality_solution_l3097_309789


namespace termites_count_workers_composition_l3097_309774

/-- The number of termites in the construction project -/
def num_termites : ℕ := 861 - 239

/-- The total number of workers in the construction project -/
def total_workers : ℕ := 861

/-- The number of monkeys in the construction project -/
def num_monkeys : ℕ := 239

/-- Theorem stating that the number of termites is 622 -/
theorem termites_count : num_termites = 622 := by
  sorry

/-- Theorem stating that the total number of workers is the sum of monkeys and termites -/
theorem workers_composition : total_workers = num_monkeys + num_termites := by
  sorry

end termites_count_workers_composition_l3097_309774


namespace orthocenter_coordinates_l3097_309716

/-- The orthocenter of a triangle --/
structure Orthocenter (A B C : ℝ × ℝ) where
  point : ℝ × ℝ
  is_orthocenter : Bool

/-- Definition of triangle ABC --/
def A : ℝ × ℝ := (5, -1)
def B : ℝ × ℝ := (4, -8)
def C : ℝ × ℝ := (-4, -4)

/-- The orthocenter of triangle ABC --/
def triangle_orthocenter : Orthocenter A B C := {
  point := (3, -5),
  is_orthocenter := sorry
}

/-- Theorem: The orthocenter of triangle ABC is (3, -5) --/
theorem orthocenter_coordinates :
  triangle_orthocenter.point = (3, -5) := by sorry

end orthocenter_coordinates_l3097_309716


namespace remainder_98_pow_24_mod_100_l3097_309740

theorem remainder_98_pow_24_mod_100 : 98^24 % 100 = 16 := by
  sorry

end remainder_98_pow_24_mod_100_l3097_309740


namespace least_bench_sections_l3097_309709

/-- Represents the capacity of a single bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Proves that 6 is the least positive integer N such that N bench sections
    can hold an equal number of adults and children -/
theorem least_bench_sections (capacity : BenchCapacity)
    (h_adults : capacity.adults = 8)
    (h_children : capacity.children = 12) :
    (∃ N : Nat, N > 0 ∧
      ∃ x : Nat, x > 0 ∧
        N * capacity.adults = x ∧
        N * capacity.children = x ∧
        ∀ M : Nat, M > 0 → M < N →
          ¬∃ y : Nat, y > 0 ∧
            M * capacity.adults = y ∧
            M * capacity.children = y) →
    (∃ N : Nat, N = 6 ∧ N > 0 ∧
      ∃ x : Nat, x > 0 ∧
        N * capacity.adults = x ∧
        N * capacity.children = x ∧
        ∀ M : Nat, M > 0 → M < N →
          ¬∃ y : Nat, y > 0 ∧
            M * capacity.adults = y ∧
            M * capacity.children = y) :=
  sorry

end least_bench_sections_l3097_309709


namespace kellys_supplies_l3097_309707

/-- Calculates the number of supplies left after Kelly's art supply shopping adventure. -/
theorem kellys_supplies (students : ℕ) (paper_per_student : ℕ) (glue_bottles : ℕ) (additional_paper : ℕ) : 
  students = 8 →
  paper_per_student = 3 →
  glue_bottles = 6 →
  additional_paper = 5 →
  ((students * paper_per_student + glue_bottles) / 2 + additional_paper : ℕ) = 20 := by
sorry

end kellys_supplies_l3097_309707


namespace monday_rainfall_value_l3097_309772

/-- The rainfall recorded over three days in centimeters -/
def total_rainfall : ℝ := 0.6666666666666666

/-- The rainfall recorded on Tuesday in centimeters -/
def tuesday_rainfall : ℝ := 0.4166666666666667

/-- The rainfall recorded on Wednesday in centimeters -/
def wednesday_rainfall : ℝ := 0.08333333333333333

/-- The rainfall recorded on Monday in centimeters -/
def monday_rainfall : ℝ := total_rainfall - (tuesday_rainfall + wednesday_rainfall)

theorem monday_rainfall_value : monday_rainfall = 0.16666666666666663 := by
  sorry

end monday_rainfall_value_l3097_309772


namespace number_with_75_halves_l3097_309733

theorem number_with_75_halves (n : ℚ) : (∃ k : ℕ, n = k * (1/2) ∧ k = 75) → n = 37.5 := by
  sorry

end number_with_75_halves_l3097_309733


namespace p_necessary_not_sufficient_for_q_l3097_309705

-- Define the conditions p and q
def p (x : ℝ) : Prop := (x - 1) * (x - 3) ≤ 0
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Define the set A satisfying condition p
def A : Set ℝ := {x | p x}

-- Define the set B satisfying condition q
def B : Set ℝ := {x | q x}

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient_for_q : 
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end p_necessary_not_sufficient_for_q_l3097_309705


namespace distance_between_points_l3097_309719

theorem distance_between_points : Real.sqrt ((8 - 2)^2 + (-5 - 3)^2) = 10 := by
  sorry

end distance_between_points_l3097_309719


namespace exists_sequence_to_1981_no_sequence_to_1982_l3097_309766

-- Define the machine operations
def multiply_by_3 (n : ℕ) : ℕ := 3 * n
def add_4 (n : ℕ) : ℕ := n + 4

-- Define a sequence of operations
inductive Operation
| Mult3 : Operation
| Add4 : Operation

def apply_operation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Mult3 => multiply_by_3 n
  | Operation.Add4 => add_4 n

def apply_sequence (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl apply_operation start

-- Theorem statements
theorem exists_sequence_to_1981 :
  ∃ (ops : List Operation), apply_sequence 1 ops = 1981 :=
sorry

theorem no_sequence_to_1982 :
  ¬∃ (ops : List Operation), apply_sequence 1 ops = 1982 :=
sorry

end exists_sequence_to_1981_no_sequence_to_1982_l3097_309766


namespace quadratic_function_inequality_l3097_309710

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0 and f(1-x) = f(1+x),
    prove that f(3^x) > f(2^x) for all x > 0 -/
theorem quadratic_function_inequality (a b c : ℝ) (x : ℝ) 
  (h1 : a > 0) 
  (h2 : ∀ y, a*(1-y)^2 + b*(1-y) + c = a*(1+y)^2 + b*(1+y) + c) 
  (h3 : x > 0) : 
  a*(3^x)^2 + b*(3^x) + c > a*(2^x)^2 + b*(2^x) + c := by
  sorry

end quadratic_function_inequality_l3097_309710


namespace fencing_required_l3097_309764

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 400 ∧ uncovered_side = 20 → 
  ∃ (width : ℝ), area = uncovered_side * width ∧ uncovered_side + 2 * width = 60 := by
  sorry

end fencing_required_l3097_309764


namespace set_properties_l3097_309785

def closed_under_transformation (A : Set ℝ) : Prop :=
  ∀ a ∈ A, (1 + a) / (1 - a) ∈ A

theorem set_properties (A : Set ℝ) (h : closed_under_transformation A) :
  (2 ∈ A → A = {2, -3, -1/2, 1/3}) ∧
  (0 ∉ A ∧ ∃ a ∈ A, A = {a, -a/(a+1), -1/(a+1), 1/(a-1)}) :=
sorry

end set_properties_l3097_309785


namespace map_to_actual_ratio_l3097_309734

-- Define the actual distance in kilometers
def actual_distance_km : ℝ := 6

-- Define the map distance in centimeters
def map_distance_cm : ℝ := 20

-- Define the conversion factor from kilometers to centimeters
def km_to_cm : ℝ := 100000

-- Theorem statement
theorem map_to_actual_ratio :
  (map_distance_cm / (actual_distance_km * km_to_cm)) = (1 / 30000) := by
  sorry

end map_to_actual_ratio_l3097_309734


namespace compare_power_towers_l3097_309757

def power_tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (power_tower base n)

theorem compare_power_towers (n : ℕ) :
  (n ≥ 3 → power_tower 3 (n - 1) > power_tower 2 n) ∧
  (n ≥ 2 → power_tower 3 n > power_tower 4 (n - 1)) :=
by sorry

end compare_power_towers_l3097_309757


namespace order_of_abc_l3097_309748

theorem order_of_abc (a b c : ℝ) (ha : a = 17/18) (hb : b = Real.cos (1/3)) (hc : c = 3 * Real.sin (1/3)) :
  c > b ∧ b > a := by
  sorry

end order_of_abc_l3097_309748


namespace fencing_required_l3097_309747

/-- Calculates the required fencing for a rectangular field -/
theorem fencing_required (area : ℝ) (side : ℝ) : 
  area = 810 ∧ side = 30 → 
  ∃ (other_side : ℝ), 
    area = side * other_side ∧ 
    side + other_side + side = 87 := by
  sorry

end fencing_required_l3097_309747


namespace sum_of_products_l3097_309721

theorem sum_of_products : 64 * 46 + 73 * 37 + 82 * 28 + 91 * 19 = 9670 := by
  sorry

end sum_of_products_l3097_309721


namespace factorization_problems_l3097_309715

theorem factorization_problems (m x y : ℝ) : 
  (m^3 - 2*m^2 - 4*m + 8 = (m-2)^2*(m+2)) ∧ 
  (x^2 - 2*x*y + y^2 - 9 = (x-y+3)*(x-y-3)) := by
  sorry

end factorization_problems_l3097_309715


namespace circle_area_and_circumference_l3097_309763

/-- Given a circle with diameter endpoints at (1,1) and (8,6), prove its area and circumference -/
theorem circle_area_and_circumference :
  let C : ℝ × ℝ := (1, 1)
  let D : ℝ × ℝ := (8, 6)
  let diameter := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let radius := diameter / 2
  let area := π * radius^2
  let circumference := 2 * π * radius
  (area = 74 * π / 4) ∧ (circumference = Real.sqrt 74 * π) := by
  sorry


end circle_area_and_circumference_l3097_309763


namespace expression_evaluation_l3097_309736

theorem expression_evaluation : -20 + 7 * ((8 - 2) / 3) = -6 := by
  sorry

end expression_evaluation_l3097_309736


namespace barbara_savings_weeks_l3097_309711

/-- Calculates the number of weeks needed to save for a wristwatch -/
def weeks_to_save (watch_cost : ℕ) (weekly_allowance : ℕ) (current_savings : ℕ) : ℕ :=
  ((watch_cost - current_savings) + weekly_allowance - 1) / weekly_allowance

/-- Proves that Barbara needs 16 more weeks to save for the watch -/
theorem barbara_savings_weeks :
  weeks_to_save 100 5 20 = 16 := by
sorry

end barbara_savings_weeks_l3097_309711


namespace x_value_proof_l3097_309723

theorem x_value_proof (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt ((10 * x) / 3) = x) : x = 10 / 3 := by
  sorry

end x_value_proof_l3097_309723


namespace max_area_triangle_line_circle_l3097_309724

/-- The maximum area of a triangle formed by the origin and two intersection points of a line and a unit circle --/
theorem max_area_triangle_line_circle : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let line (k : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 - 1}
  let intersectionPoints (k : ℝ) := circle ∩ line k
  let triangleArea (A B : ℝ × ℝ) := (1/2) * abs (A.1 * B.2 - A.2 * B.1)
  ∀ k : ℝ, ∀ A B : ℝ × ℝ, A ∈ intersectionPoints k → B ∈ intersectionPoints k → A ≠ B →
    triangleArea A B ≤ 1/2 :=
by sorry

end max_area_triangle_line_circle_l3097_309724


namespace fraction_equals_44_l3097_309760

theorem fraction_equals_44 : (2450 - 2377)^2 / 121 = 44 := by
  sorry

end fraction_equals_44_l3097_309760


namespace infinitely_many_m_for_binomial_equality_l3097_309778

theorem infinitely_many_m_for_binomial_equality :
  ∀ n : ℕ, n ≥ 4 →
  ∃ m : ℕ, m ≥ 2 ∧
    m = (n^2 - 3*n + 2) / 2 ∧
    Nat.choose m 2 = 3 * Nat.choose n 4 :=
by sorry

end infinitely_many_m_for_binomial_equality_l3097_309778


namespace sqrt_eight_times_sqrt_two_l3097_309777

theorem sqrt_eight_times_sqrt_two : Real.sqrt 8 * Real.sqrt 2 = 4 := by
  sorry

end sqrt_eight_times_sqrt_two_l3097_309777


namespace set_inclusion_l3097_309701

-- Define the sets M, N, and P
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + 
                             Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

-- State the theorem
theorem set_inclusion : M ⊆ P ∧ P ⊆ N := by sorry

end set_inclusion_l3097_309701


namespace sequence_length_l3097_309795

theorem sequence_length (a₁ : ℕ) (aₙ : ℕ) (d : ℤ) (n : ℕ) :
  a₁ = 150 ∧ aₙ = 30 ∧ d = -6 →
  n = 21 ∧ aₙ = a₁ + d * (n - 1) :=
by sorry

end sequence_length_l3097_309795


namespace correct_sampling_methods_l3097_309713

-- Define the sampling scenarios
structure SamplingScenario where
  total : ℕ
  sample_size : ℕ
  categories : Option (List ℕ)

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

-- Define the function to determine the most appropriate sampling method
def most_appropriate_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

-- Theorem to prove the correct sampling methods for given scenarios
theorem correct_sampling_methods :
  let scenario1 := SamplingScenario.mk 10 2 none
  let scenario2 := SamplingScenario.mk 1920 32 none
  let scenario3 := SamplingScenario.mk 160 20 (some [120, 16, 24])
  (most_appropriate_sampling_method scenario1 = SamplingMethod.SimpleRandom) ∧
  (most_appropriate_sampling_method scenario2 = SamplingMethod.Systematic) ∧
  (most_appropriate_sampling_method scenario3 = SamplingMethod.Stratified) :=
  sorry

end correct_sampling_methods_l3097_309713


namespace line_through_point_l3097_309752

/-- Given a line equation 2bx + (b+2)y = b + 6 that passes through the point (-3, 4), prove that b = 2/3 -/
theorem line_through_point (b : ℝ) : 
  (2 * b * (-3) + (b + 2) * 4 = b + 6) → b = 2/3 := by
  sorry

end line_through_point_l3097_309752


namespace carpet_transformation_l3097_309728

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a piece cut from a rectangle -/
structure CutPiece where
  width : ℕ
  height : ℕ

/-- Represents the original carpet -/
def original_carpet : Rectangle := { width := 9, height := 12 }

/-- Represents the piece cut off by the dragon -/
def dragon_cut : CutPiece := { width := 1, height := 8 }

/-- Represents the final square carpet -/
def final_carpet : Rectangle := { width := 10, height := 10 }

/-- Function to calculate the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- Function to calculate the area of a cut piece -/
def cut_area (c : CutPiece) : ℕ := c.width * c.height

/-- Theorem stating that it's possible to transform the damaged carpet into a square -/
theorem carpet_transformation :
  ∃ (part1 part2 part3 : Rectangle),
    area original_carpet - cut_area dragon_cut =
    area part1 + area part2 + area part3 ∧
    area final_carpet = area part1 + area part2 + area part3 := by
  sorry

end carpet_transformation_l3097_309728


namespace smallest_prime_divisor_of_sum_l3097_309796

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^19 + 11^13) ∧ ∀ q, Nat.Prime q → q ∣ (3^19 + 11^13) → p ≤ q :=
by sorry

end smallest_prime_divisor_of_sum_l3097_309796


namespace norm_scalar_multiple_l3097_309792

theorem norm_scalar_multiple {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] 
  (v : V) (h : ‖v‖ = 5) : ‖(4 : ℝ) • v‖ = 20 := by
  sorry

end norm_scalar_multiple_l3097_309792


namespace min_value_expression_l3097_309708

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (4 * q) / (2 * p + 2 * r) ≥ 5 / 2 := by
  sorry

end min_value_expression_l3097_309708


namespace larger_number_problem_l3097_309744

theorem larger_number_problem (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := by
  sorry

end larger_number_problem_l3097_309744


namespace square_area_8m_l3097_309787

theorem square_area_8m (side_length : ℝ) (area : ℝ) : 
  side_length = 8 → area = side_length ^ 2 → area = 64 := by sorry

end square_area_8m_l3097_309787


namespace course_selection_methods_l3097_309729

theorem course_selection_methods (n : ℕ) (k : ℕ) : 
  n = 3 → k = 4 → n ^ k = 81 := by sorry

end course_selection_methods_l3097_309729


namespace events_mutually_exclusive_and_complementary_l3097_309714

/-- Represents the number of male students in the group -/
def num_male : ℕ := 3

/-- Represents the number of female students in the group -/
def num_female : ℕ := 2

/-- Represents the number of students selected for the competition -/
def num_selected : ℕ := 2

/-- Represents the event of selecting at least one female student -/
def at_least_one_female : Set (Fin num_male × Fin num_female) := sorry

/-- Represents the event of selecting all male students -/
def all_male : Set (Fin num_male × Fin num_female) := sorry

/-- Theorem stating that the events are mutually exclusive and complementary -/
theorem events_mutually_exclusive_and_complementary :
  (at_least_one_female ∩ all_male = ∅) ∧
  (at_least_one_female ∪ all_male = Set.univ) :=
sorry

end events_mutually_exclusive_and_complementary_l3097_309714


namespace vectors_linearly_dependent_iff_l3097_309739

/-- Two vectors in ℝ² -/
def v1 : Fin 2 → ℝ := ![2, 5]
def v2 (m : ℝ) : Fin 2 → ℝ := ![4, m]

/-- Definition of linear dependence for two vectors -/
def linearlyDependent (u v : Fin 2 → ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (∀ i, a * u i + b * v i = 0)

/-- Theorem: The vectors v1 and v2 are linearly dependent iff m = 10 -/
theorem vectors_linearly_dependent_iff (m : ℝ) :
  linearlyDependent v1 (v2 m) ↔ m = 10 := by
  sorry

end vectors_linearly_dependent_iff_l3097_309739


namespace probability_cap_given_sunglasses_l3097_309722

/-- The number of people wearing sunglasses -/
def sunglasses_wearers : ℕ := 60

/-- The number of people wearing caps -/
def cap_wearers : ℕ := 40

/-- The number of people wearing both sunglasses and caps and hats -/
def triple_wearers : ℕ := 8

/-- The probability that a person wearing a cap is also wearing sunglasses -/
def prob_sunglasses_given_cap : ℚ := 1/2

theorem probability_cap_given_sunglasses :
  let both_wearers := cap_wearers * prob_sunglasses_given_cap
  (both_wearers : ℚ) / sunglasses_wearers = 1/3 := by sorry

end probability_cap_given_sunglasses_l3097_309722


namespace opposite_numbers_expression_l3097_309761

theorem opposite_numbers_expression (m n : ℝ) (h : m + n = 0) :
  3 * (m - n) - (1/2) * (2 * m - 10 * n) = 0 := by
  sorry

end opposite_numbers_expression_l3097_309761


namespace smallest_non_negative_solution_l3097_309727

theorem smallest_non_negative_solution (x : ℕ) : x = 2 ↔ 
  (∀ y : ℕ, (42 * y + 10) % 15 = 5 → y ≥ x) ∧ (42 * x + 10) % 15 = 5 := by
  sorry

end smallest_non_negative_solution_l3097_309727


namespace tangerine_count_l3097_309788

theorem tangerine_count (apples pears tangerines : ℕ) : 
  apples = 45 →
  apples = pears + 21 →
  tangerines = pears + 18 →
  tangerines = 42 := by
sorry

end tangerine_count_l3097_309788


namespace permutations_of_eight_distinct_objects_l3097_309743

theorem permutations_of_eight_distinct_objects : Nat.factorial 8 = 40320 := by
  sorry

end permutations_of_eight_distinct_objects_l3097_309743


namespace profit_margin_calculation_l3097_309786

/-- Profit margin calculation -/
theorem profit_margin_calculation (n : ℝ) (C S M : ℝ) 
  (h1 : M = (1 / n) * (2 * C - S)) 
  (h2 : S - M = C) : 
  M = S / (n + 2) := by
  sorry

end profit_margin_calculation_l3097_309786


namespace square_from_triangles_even_count_l3097_309753

-- Define the triangle type
structure Triangle :=
  (side1 : ℕ)
  (side2 : ℕ)
  (side3 : ℕ)

-- Define the properties of our specific triangle
def SpecificTriangle : Triangle :=
  { side1 := 3, side2 := 4, side3 := 5 }

-- Define the area of the triangle
def triangleArea (t : Triangle) : ℚ :=
  (t.side1 * t.side2 : ℚ) / 2

-- Define a function to check if a number is even
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Main theorem
theorem square_from_triangles_even_count :
  ∀ n : ℕ, n > 0 →
  (∃ a : ℕ, a > 0 ∧ (a : ℚ)^2 = n * triangleArea SpecificTriangle) →
  isEven n :=
sorry

end square_from_triangles_even_count_l3097_309753


namespace twentieth_common_number_l3097_309765

/-- The mth term of the first sequence -/
def a (m : ℕ) : ℕ := 4 * m - 1

/-- The nth term of the second sequence -/
def b (n : ℕ) : ℕ := 3 * n + 2

/-- The kth common number between the two sequences -/
def common_number (k : ℕ) : ℕ := 12 * k - 1

theorem twentieth_common_number :
  ∃ m n : ℕ, a m = b n ∧ a m = common_number 20 ∧ common_number 20 = 239 := by
  sorry

end twentieth_common_number_l3097_309765


namespace certain_number_proof_l3097_309770

theorem certain_number_proof : ∃ (n : ℕ), n + 3327 = 13200 ∧ n = 9873 := by
  sorry

end certain_number_proof_l3097_309770


namespace stock_value_after_fluctuations_l3097_309749

theorem stock_value_after_fluctuations (initial_value : ℝ) (initial_value_pos : initial_value > 0) :
  let limit_up := 1.1
  let limit_down := 0.9
  let final_value := initial_value * (limit_up ^ 5) * (limit_down ^ 5)
  final_value < initial_value :=
by sorry

end stock_value_after_fluctuations_l3097_309749


namespace rhombus_side_length_l3097_309704

/-- 
A rhombus is a quadrilateral with four equal sides.
The perimeter of a rhombus is the sum of the lengths of all four sides.
-/
structure Rhombus where
  side_length : ℝ
  perimeter : ℝ
  perimeter_eq : perimeter = 4 * side_length

theorem rhombus_side_length (r : Rhombus) (h : r.perimeter = 4) : r.side_length = 1 := by
  sorry

end rhombus_side_length_l3097_309704


namespace vector_relations_l3097_309741

/-- Given vectors a, b, n, and c in ℝ², prove the values of k for perpendicularity and parallelism conditions. -/
theorem vector_relations (a b n c : ℝ × ℝ) (k : ℝ) : 
  a = (-3, 1) → 
  b = (1, -2) → 
  c = (1, -1) → 
  n = (a.1 + k * b.1, a.2 + k * b.2) → 
  (((n.1 * (2 * a.1 - b.1) + n.2 * (2 * a.2 - b.2) = 0) → k = 5/3) ∧ 
   ((n.1 * (c.1 + k * b.1) = n.2 * (c.2 + k * b.2)) → k = -1/3)) := by
  sorry


end vector_relations_l3097_309741


namespace tournament_matches_divisible_by_seven_l3097_309745

/-- Represents a single elimination tournament --/
structure Tournament :=
  (total_players : ℕ)
  (bye_players : ℕ)

/-- Calculates the total number of matches in a tournament --/
def total_matches (t : Tournament) : ℕ := t.total_players - 1

/-- Theorem: In a tournament with 120 players and 32 byes, the total matches is divisible by 7 --/
theorem tournament_matches_divisible_by_seven :
  ∀ (t : Tournament), t.total_players = 120 → t.bye_players = 32 →
  ∃ (k : ℕ), total_matches t = 7 * k := by
  sorry

end tournament_matches_divisible_by_seven_l3097_309745


namespace min_value_of_fraction_sum_min_value_achieved_l3097_309758

theorem min_value_of_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ (1 / a₀ + 2 / b₀ = 3 + 2 * Real.sqrt 2) := by
  sorry

end min_value_of_fraction_sum_min_value_achieved_l3097_309758


namespace birds_and_storks_on_fence_l3097_309790

theorem birds_and_storks_on_fence (initial_birds : ℕ) (initial_storks : ℕ) (new_birds : ℕ) : 
  initial_birds = 3 → initial_storks = 2 → new_birds = 5 →
  initial_birds + initial_storks + new_birds = 10 := by
  sorry

end birds_and_storks_on_fence_l3097_309790
