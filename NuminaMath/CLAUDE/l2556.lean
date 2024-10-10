import Mathlib

namespace bruce_bought_five_crayons_l2556_255624

/-- Calculates the number of packs of crayons Bruce bought given the conditions of the problem. -/
def bruces_crayons (crayonPrice bookPrice calculatorPrice bagPrice totalMoney : ℕ) 
  (numBooks numCalculators numBags : ℕ) : ℕ :=
  let bookCost := numBooks * bookPrice
  let calculatorCost := numCalculators * calculatorPrice
  let bagCost := numBags * bagPrice
  let remainingMoney := totalMoney - bookCost - calculatorCost - bagCost
  remainingMoney / crayonPrice

/-- Theorem stating that Bruce bought 5 packs of crayons given the conditions of the problem. -/
theorem bruce_bought_five_crayons : 
  bruces_crayons 5 5 5 10 200 10 3 11 = 5 := by
  sorry

end bruce_bought_five_crayons_l2556_255624


namespace interest_rate_satisfies_conditions_interest_rate_unique_solution_l2556_255630

/-- The principal amount -/
def P : ℝ := 6800.000000000145

/-- The time period in years -/
def t : ℝ := 2

/-- The difference between compound interest and simple interest -/
def diff : ℝ := 17

/-- The interest rate as a percentage -/
def r : ℝ := 5

/-- Theorem stating that the given interest rate satisfies the conditions -/
theorem interest_rate_satisfies_conditions :
  P * (1 + r / 100) ^ t - P - (P * r * t / 100) = diff := by
  sorry

/-- Theorem stating that the given interest rate is the unique solution -/
theorem interest_rate_unique_solution :
  ∀ x : ℝ, P * (1 + x / 100) ^ t - P - (P * x * t / 100) = diff → x = r := by
  sorry

end interest_rate_satisfies_conditions_interest_rate_unique_solution_l2556_255630


namespace conference_handshakes_l2556_255686

/-- The number of handshakes in a conference with multiple companies --/
def num_handshakes (num_companies : ℕ) (representatives_per_company : ℕ) : ℕ :=
  let total_people := num_companies * representatives_per_company
  let handshakes_per_person := total_people - representatives_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a conference with 5 companies, each having 5 representatives,
    where every person shakes hands once with every person except those from
    their own company, the total number of handshakes is 250. --/
theorem conference_handshakes :
  num_handshakes 5 5 = 250 := by
  sorry

end conference_handshakes_l2556_255686


namespace smallest_n_remainder_l2556_255659

theorem smallest_n_remainder (N : ℕ) : 
  (N > 0) →
  (∃ k : ℕ, 2008 * N = k^2) →
  (∃ m : ℕ, 2007 * N = m^3) →
  (∀ M : ℕ, M < N → (¬∃ k : ℕ, 2008 * M = k^2) ∨ (¬∃ m : ℕ, 2007 * M = m^3)) →
  N % 25 = 17 := by
sorry

end smallest_n_remainder_l2556_255659


namespace quadratic_root_condition_l2556_255603

theorem quadratic_root_condition (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (∀ x : ℝ, x^2 + p*x + p - 1 = 0 ↔ x = x₁ ∨ x = x₂) ∧ 
    x₁^2 + x₁^3 = -(x₂^2 + x₂^3)) ↔ 
  p = 1 ∨ p = 2 := by sorry

end quadratic_root_condition_l2556_255603


namespace ball_probabilities_l2556_255649

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  red : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (counts : BallCounts) : ℕ :=
  counts.red + counts.yellow + counts.white

/-- Calculates the probability of picking a ball of a specific color -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  (favorable : ℚ) / (total : ℚ)

theorem ball_probabilities (initial : BallCounts)
    (h_initial : initial = ⟨10, 6, 4⟩) :
  let total := totalBalls initial
  (probability initial.white total = 1/5) ∧
  (probability (initial.red + initial.yellow) total = 4/5) ∧
  (probability (initial.white - 2) (total - 4) = 1/8) := by
  sorry

end ball_probabilities_l2556_255649


namespace root_sum_absolute_value_l2556_255674

theorem root_sum_absolute_value (m : ℝ) (α β : ℝ) 
  (h1 : α^2 - 22*α + m = 0)
  (h2 : β^2 - 22*β + m = 0)
  (h3 : m ≤ 121) : 
  |α| + |β| = if 0 ≤ m then 22 else Real.sqrt (484 - 4*m) :=
by sorry

end root_sum_absolute_value_l2556_255674


namespace circle_center_l2556_255627

/-- The equation of a circle in the form x^2 - 6x + y^2 + 2y = 9 -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y = 9

/-- The center of a circle -/
def CircleCenter (h k : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x - h)^2 + (y - k)^2 = 19

/-- Theorem: The center of the circle with equation x^2 - 6x + y^2 + 2y = 9 is (3, -1) -/
theorem circle_center :
  CircleCenter 3 (-1) CircleEquation :=
sorry

end circle_center_l2556_255627


namespace product_of_symmetric_complex_numbers_l2556_255669

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem product_of_symmetric_complex_numbers :
  ∀ z₁ z₂ : ℂ, symmetric_wrt_imaginary_axis z₁ z₂ → z₁ = 2 + I → z₁ * z₂ = -5 := by
  sorry

#check product_of_symmetric_complex_numbers

end product_of_symmetric_complex_numbers_l2556_255669


namespace slope_of_line_l2556_255618

theorem slope_of_line (x y : ℝ) : y = x - 1 → (y - (x - 1)) / (x - x) = 1 := by
  sorry

end slope_of_line_l2556_255618


namespace right_triangle_acute_angle_l2556_255656

theorem right_triangle_acute_angle (α : ℝ) : 
  α > 0 ∧ α < 90 → -- α is an acute angle
  α + (α - 10) + 90 = 180 → -- sum of angles in the triangle
  α = 50 := by
sorry

end right_triangle_acute_angle_l2556_255656


namespace modulo_eleven_residue_l2556_255616

theorem modulo_eleven_residue : (368 + 3 * 78 + 8 * 242 + 6 * 22) % 11 = 8 := by
  sorry

end modulo_eleven_residue_l2556_255616


namespace fraction_comparison_l2556_255641

theorem fraction_comparison (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 3 →
  (5 * x + 2 > 8 - 3 * x) ↔ (x ∈ Set.Ioo (3/4 : ℝ) 3) :=
by sorry

end fraction_comparison_l2556_255641


namespace complement_union_is_empty_l2556_255650

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {2, 4, 5}

theorem complement_union_is_empty :
  (U \ (M ∪ N)) = ∅ := by
  sorry

end complement_union_is_empty_l2556_255650


namespace mixed_number_calculation_l2556_255631

theorem mixed_number_calculation : 
  (7 + 1/2 - (5 + 3/4)) * (3 + 1/6 + 2 + 1/8) = 9 + 25/96 := by
  sorry

end mixed_number_calculation_l2556_255631


namespace select_students_l2556_255638

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem select_students (num_boys num_girls : ℕ) (boys_selected girls_selected : ℕ) : 
  num_boys = 11 → num_girls = 10 → boys_selected = 2 → girls_selected = 3 →
  (choose num_girls girls_selected) * (choose num_boys boys_selected) = 6600 := by
  sorry

end select_students_l2556_255638


namespace largest_square_from_rectangle_l2556_255640

theorem largest_square_from_rectangle (width length : ℕ) 
  (h_width : width = 32) (h_length : length = 74) :
  ∃ (side : ℕ), side = Nat.gcd width length ∧ 
  side * (width / side) = width ∧ 
  side * (length / side) = length ∧
  ∀ (n : ℕ), n * (width / n) = width ∧ n * (length / n) = length → n ≤ side :=
sorry

end largest_square_from_rectangle_l2556_255640


namespace function_properties_l2556_255689

noncomputable section

variable (a : ℝ)
variable (f : ℝ → ℝ)

def has_extremum_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a b, ∀ y ∈ Set.Ioo a b, f x ≤ f y ∨ f x ≥ f y

def has_unique_zero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

theorem function_properties (h1 : a > 0) 
    (h2 : f = λ x => Real.exp (2*x) + 2 / Real.exp x - a*x) :
  (has_extremum_in f 0 1 → 0 < a ∧ a < 2 * Real.exp 2 - 2 / Real.exp 1) ∧
  (has_unique_zero f → ∃ x₀, f x₀ = 0 ∧ Real.log 2 < x₀ ∧ x₀ < 1) := by
  sorry

end

end function_properties_l2556_255689


namespace toilet_paper_packs_is_14_l2556_255699

/-- The number of packs of toilet paper Stella needs to buy after 4 weeks -/
def toilet_paper_packs : ℕ :=
  let bathrooms : ℕ := 6
  let days_per_week : ℕ := 7
  let rolls_per_pack : ℕ := 12
  let weeks : ℕ := 4
  let rolls_per_day : ℕ := bathrooms
  let rolls_per_week : ℕ := rolls_per_day * days_per_week
  let total_rolls : ℕ := rolls_per_week * weeks
  total_rolls / rolls_per_pack

theorem toilet_paper_packs_is_14 : toilet_paper_packs = 14 := by
  sorry

end toilet_paper_packs_is_14_l2556_255699


namespace james_car_rental_days_l2556_255644

/-- Calculates the number of days James rents his car per week -/
def days_rented_per_week (hourly_rate : ℕ) (hours_per_day : ℕ) (weekly_earnings : ℕ) : ℕ :=
  weekly_earnings / (hourly_rate * hours_per_day)

/-- Theorem stating that James rents his car for 4 days per week -/
theorem james_car_rental_days :
  days_rented_per_week 20 8 640 = 4 := by
  sorry

end james_car_rental_days_l2556_255644


namespace ship_length_l2556_255654

theorem ship_length (emily_step : ℝ) (ship_step : ℝ) :
  let emily_forward := 150
  let emily_backward := 70
  let wind_factor := 0.9
  let ship_length := 150 * emily_step - 150 * ship_step
  emily_backward * emily_step = ship_length - emily_backward * ship_step * wind_factor
  →
  ship_length = 19950 / 213 * emily_step :=
by sorry

end ship_length_l2556_255654


namespace billy_ate_nine_apples_on_wednesday_l2556_255637

/-- The number of apples Billy ate each day of the week --/
structure WeeklyApples where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  total : ℕ

/-- Billy's apple consumption for the week satisfies the given conditions --/
def satisfiesConditions (w : WeeklyApples) : Prop :=
  w.monday = 2 ∧
  w.tuesday = 2 * w.monday ∧
  w.thursday = 4 * w.friday ∧
  w.friday = w.monday / 2 ∧
  w.total = 20 ∧
  w.total = w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- The theorem stating that Billy ate 9 apples on Wednesday --/
theorem billy_ate_nine_apples_on_wednesday (w : WeeklyApples) 
  (h : satisfiesConditions w) : w.wednesday = 9 := by
  sorry

end billy_ate_nine_apples_on_wednesday_l2556_255637


namespace circles_intersect_l2556_255673

/-- Two circles in a 2D plane --/
structure TwoCircles where
  c1_center : ℝ × ℝ
  c2_center : ℝ × ℝ
  c1_radius : ℝ
  c2_radius : ℝ

/-- Definition of intersecting circles --/
def are_intersecting (tc : TwoCircles) : Prop :=
  let d := Real.sqrt ((tc.c2_center.1 - tc.c1_center.1)^2 + (tc.c2_center.2 - tc.c1_center.2)^2)
  (tc.c1_radius + tc.c2_radius > d) ∧ (d > abs (tc.c1_radius - tc.c2_radius))

/-- The main theorem --/
theorem circles_intersect (tc : TwoCircles) 
  (h1 : tc.c1_center = (-2, 2))
  (h2 : tc.c2_center = (2, 5))
  (h3 : tc.c1_radius = 2)
  (h4 : tc.c2_radius = 4)
  (h5 : Real.sqrt ((tc.c2_center.1 - tc.c1_center.1)^2 + (tc.c2_center.2 - tc.c1_center.2)^2) = 5)
  : are_intersecting tc := by
  sorry

end circles_intersect_l2556_255673


namespace loan_amount_proof_l2556_255658

/-- Represents the interest rate as a decimal -/
def interest_rate : ℝ := 0.04

/-- Represents the loan duration in years -/
def years : ℕ := 2

/-- Calculates the compound interest amount after n years -/
def compound_interest (P : ℝ) : ℝ := P * (1 + interest_rate) ^ years

/-- Calculates the simple interest amount after n years -/
def simple_interest (P : ℝ) : ℝ := P * (1 + interest_rate * years)

/-- The difference between compound and simple interest -/
def interest_difference : ℝ := 10.40

theorem loan_amount_proof (P : ℝ) : 
  compound_interest P - simple_interest P = interest_difference → P = 6500 := by
  sorry

end loan_amount_proof_l2556_255658


namespace function_f_is_identity_l2556_255653

/-- A function satisfying the given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

/-- Theorem stating that the only function satisfying the conditions is the identity function -/
theorem function_f_is_identity (f : ℝ → ℝ) (hf : FunctionF f) : 
  ∀ x : ℝ, f x = x := by
  sorry

end function_f_is_identity_l2556_255653


namespace triangle_square_distance_l2556_255648

-- Define the triangle ABF
def Triangle (A B F : ℝ × ℝ) : Prop :=
  ∃ (x y z : ℝ), 
    (A.1 - F.1)^2 + (A.2 - F.2)^2 = x^2 ∧
    (B.1 - F.1)^2 + (B.2 - F.2)^2 = y^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = z^2

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  ∃ (s : ℝ),
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = s^2 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = s^2 ∧
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = s^2 ∧
    (A.1 - D.1)^2 + (A.2 - D.2)^2 = s^2

-- Define the circumcenter of a square
def Circumcenter (E A B C D : ℝ × ℝ) : Prop :=
  (E.1 - A.1)^2 + (E.2 - A.2)^2 = (E.1 - B.1)^2 + (E.2 - B.2)^2 ∧
  (E.1 - B.1)^2 + (E.2 - B.2)^2 = (E.1 - C.1)^2 + (E.2 - C.2)^2 ∧
  (E.1 - C.1)^2 + (E.2 - C.2)^2 = (E.1 - D.1)^2 + (E.2 - D.2)^2

theorem triangle_square_distance 
  (A B C D E F : ℝ × ℝ)
  (h1 : Triangle A B F)
  (h2 : Square A B C D)
  (h3 : Circumcenter E A B C D)
  (h4 : (A.1 - F.1)^2 + (A.2 - F.2)^2 = 36)
  (h5 : (B.1 - F.1)^2 + (B.2 - F.2)^2 = 64)
  (h6 : (A.1 - B.1) * (F.1 - B.1) + (A.2 - B.2) * (F.2 - B.2) = 0) :
  (E.1 - F.1)^2 + (E.2 - F.2)^2 = 98 :=
by sorry

end triangle_square_distance_l2556_255648


namespace laundry_charge_calculation_l2556_255636

/-- The amount charged per kilo of laundry -/
def charge_per_kilo : ℝ := sorry

/-- The number of kilos washed two days ago -/
def kilos_two_days_ago : ℝ := 5

/-- The number of kilos washed yesterday -/
def kilos_yesterday : ℝ := kilos_two_days_ago + 5

/-- The number of kilos washed today -/
def kilos_today : ℝ := 2 * kilos_yesterday

/-- The total earnings for three days -/
def total_earnings : ℝ := 70

theorem laundry_charge_calculation :
  charge_per_kilo * (kilos_two_days_ago + kilos_yesterday + kilos_today) = total_earnings ∧
  charge_per_kilo = 2 := by sorry

end laundry_charge_calculation_l2556_255636


namespace restaurant_glasses_count_l2556_255671

theorem restaurant_glasses_count :
  -- Define the number of glasses in each box type
  let small_box_glasses : ℕ := 12
  let large_box_glasses : ℕ := 16
  -- Define the difference in number of boxes
  let box_difference : ℕ := 16
  -- Define the average number of glasses per box
  let average_glasses : ℚ := 15
  -- Define variables for the number of each type of box
  ∀ (small_boxes large_boxes : ℕ),
  -- Condition: There are 16 more large boxes than small boxes
  large_boxes = small_boxes + box_difference →
  -- Condition: The average number of glasses per box is 15
  (small_box_glasses * small_boxes + large_box_glasses * large_boxes : ℚ) / 
    (small_boxes + large_boxes : ℚ) = average_glasses →
  -- Conclusion: The total number of glasses is 480
  small_box_glasses * small_boxes + large_box_glasses * large_boxes = 480 :=
by
  sorry

end restaurant_glasses_count_l2556_255671


namespace binomial_12_11_l2556_255680

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binomial_12_11_l2556_255680


namespace video_game_players_l2556_255643

/-- The number of players who joined a video game -/
def players_joined : ℕ := 5

theorem video_game_players :
  let initial_players : ℕ := 4
  let lives_per_player : ℕ := 3
  let total_lives : ℕ := 27
  players_joined = (total_lives - initial_players * lives_per_player) / lives_per_player :=
by
  sorry

#check video_game_players

end video_game_players_l2556_255643


namespace problem_solution_l2556_255646

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 4*a| + |x|

-- Theorem statement
theorem problem_solution :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ a^2) →
  (-4 ≤ a ∧ a ≤ 4) ∧
  (∃ min_value : ℝ, min_value = 16/21 ∧
    ∀ x y z : ℝ, 4*x + 2*y + z = 4 →
      (x + y)^2 + y^2 + z^2 ≥ min_value) :=
by sorry

end problem_solution_l2556_255646


namespace class_selection_combinations_l2556_255621

-- Define the number of total classes and classes to choose
def n : ℕ := 10
def r : ℕ := 4

-- Define the combination function
def combination (n r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

-- State the theorem
theorem class_selection_combinations : combination n r = 210 := by
  sorry

end class_selection_combinations_l2556_255621


namespace max_value_function_l2556_255635

theorem max_value_function (t : ℝ) : (3^t - 4*t)*t / (9^t + t) ≤ 1/16 := by
  sorry

end max_value_function_l2556_255635


namespace cubic_sum_minus_product_l2556_255642

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by
sorry

end cubic_sum_minus_product_l2556_255642


namespace smallest_positive_integers_difference_l2556_255666

def m : ℕ := sorry

def n : ℕ := sorry

theorem smallest_positive_integers_difference : 
  (m ≥ 100) ∧ 
  (m < 1000) ∧ 
  (m % 13 = 6) ∧ 
  (∀ k : ℕ, k ≥ 100 ∧ k < 1000 ∧ k % 13 = 6 → m ≤ k) ∧
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 17 = 7) ∧ 
  (∀ l : ℕ, l ≥ 1000 ∧ l < 10000 ∧ l % 17 = 7 → n ≤ l) →
  n - m = 900 := by sorry

end smallest_positive_integers_difference_l2556_255666


namespace max_d_value_l2556_255609

/-- Represents a 6-digit number of the form 6d6,33f -/
def sixDigitNumber (d f : ℕ) : ℕ := 600000 + 10000*d + 3300 + f

/-- Predicate for d and f being single digits -/
def areSingleDigits (d f : ℕ) : Prop := d < 10 ∧ f < 10

/-- Predicate for the number being divisible by 33 -/
def isDivisibleBy33 (d f : ℕ) : Prop :=
  (sixDigitNumber d f) % 33 = 0

theorem max_d_value :
  ∃ (d : ℕ), 
    (∃ (f : ℕ), areSingleDigits d f ∧ isDivisibleBy33 d f) ∧
    (∀ (d' f' : ℕ), areSingleDigits d' f' → isDivisibleBy33 d' f' → d' ≤ d) ∧
    d = 1 :=
sorry

end max_d_value_l2556_255609


namespace probability_standard_deck_l2556_255697

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- Probability of drawing two red cards followed by two black cards -/
def probability_two_red_two_black (d : Deck) : Rat :=
  if d.total_cards ≥ 4 ∧ d.red_cards ≥ 2 ∧ d.black_cards ≥ 2 then
    (d.red_cards * (d.red_cards - 1) * d.black_cards * (d.black_cards - 1)) /
    (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2) * (d.total_cards - 3))
  else
    0

theorem probability_standard_deck :
  probability_two_red_two_black ⟨52, 26, 26⟩ = 325 / 4998 := by
  sorry

end probability_standard_deck_l2556_255697


namespace calculation_proof_l2556_255612

theorem calculation_proof : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end calculation_proof_l2556_255612


namespace new_plant_characteristics_l2556_255620

/-- Represents a plant with genetic characteristics -/
structure Plant where
  ploidy : Nat
  has_homologous_chromosomes : Bool
  can_form_fertile_gametes : Bool
  homozygosity : Option Bool

/-- Represents the process of obtaining new plants from treated corn -/
def obtain_new_plants (original : Plant) (colchicine_treated : Bool) (anther_culture : Bool) : Plant :=
  sorry

/-- Theorem stating the characteristics of new plants obtained from treated corn -/
theorem new_plant_characteristics 
  (original : Plant)
  (h_original_diploid : original.ploidy = 2)
  (h_colchicine_treated : Bool)
  (h_anther_culture : Bool) :
  let new_plant := obtain_new_plants original h_colchicine_treated h_anther_culture
  new_plant.ploidy = 1 ∧ 
  new_plant.has_homologous_chromosomes = true ∧
  new_plant.can_form_fertile_gametes = true ∧
  new_plant.homozygosity = none :=
by sorry

end new_plant_characteristics_l2556_255620


namespace no_power_of_three_and_five_l2556_255647

def sequence_v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * sequence_v (n + 1) - sequence_v n

theorem no_power_of_three_and_five :
  ∀ n : ℕ, ¬∃ (a b : ℕ+), sequence_v n = 3^(a:ℕ) * 5^(b:ℕ) := by
  sorry

end no_power_of_three_and_five_l2556_255647


namespace heartsuit_four_six_l2556_255682

-- Define the ♡ operation
def heartsuit (x y : ℝ) : ℝ := 5*x + 3*y

-- Theorem statement
theorem heartsuit_four_six : heartsuit 4 6 = 38 := by
  sorry

end heartsuit_four_six_l2556_255682


namespace absolute_value_inequality_l2556_255695

theorem absolute_value_inequality (x : ℝ) :
  |2 * x + 6| < 10 ↔ -8 < x ∧ x < 2 := by
  sorry

end absolute_value_inequality_l2556_255695


namespace partnership_annual_gain_l2556_255694

/-- Represents the annual gain of a partnership given the following conditions:
    - A invests x at the beginning of the year
    - B invests 2x after 6 months
    - C invests 3x after 8 months
    - A's share is 6200
    - Profit is divided based on investment amount and time
-/
theorem partnership_annual_gain (x : ℝ) (total_gain : ℝ) : 
  x > 0 →
  (x * 12) / (x * 12 + 2 * x * 6 + 3 * x * 4) = 6200 / total_gain →
  total_gain = 18600 := by
  sorry

#check partnership_annual_gain

end partnership_annual_gain_l2556_255694


namespace total_chairs_moved_l2556_255667

/-- The total number of chairs agreed to be moved is equal to the sum of
    chairs moved by Carey, chairs moved by Pat, and chairs left to move. -/
theorem total_chairs_moved (carey_chairs pat_chairs left_chairs : ℕ)
  (h1 : carey_chairs = 28)
  (h2 : pat_chairs = 29)
  (h3 : left_chairs = 17) :
  carey_chairs + pat_chairs + left_chairs = 74 := by
  sorry

end total_chairs_moved_l2556_255667


namespace circle_theorem_sphere_theorem_l2556_255651

-- Define a circle and a sphere
def Circle : Type := Unit
def Sphere : Type := Unit

-- Define a point on a circle and a sphere
def PointOnCircle : Type := Unit
def PointOnSphere : Type := Unit

-- Define a semicircle and a hemisphere
def Semicircle : Type := Unit
def Hemisphere : Type := Unit

-- Define a function to check if a point is in a semicircle or hemisphere
def isIn : PointOnCircle → Semicircle → Prop := sorry
def isInHemisphere : PointOnSphere → Hemisphere → Prop := sorry

-- Theorem for the circle problem
theorem circle_theorem (c : Circle) (p1 p2 p3 p4 : PointOnCircle) :
  ∃ (s : Semicircle), (isIn p1 s ∧ isIn p2 s ∧ isIn p3 s) ∨
                      (isIn p1 s ∧ isIn p2 s ∧ isIn p4 s) ∨
                      (isIn p1 s ∧ isIn p3 s ∧ isIn p4 s) ∨
                      (isIn p2 s ∧ isIn p3 s ∧ isIn p4 s) :=
sorry

-- Theorem for the sphere problem
theorem sphere_theorem (s : Sphere) (p1 p2 p3 p4 p5 : PointOnSphere) :
  ∃ (h : Hemisphere), (isInHemisphere p1 h ∧ isInHemisphere p2 h ∧ isInHemisphere p3 h ∧ isInHemisphere p4 h) ∨
                      (isInHemisphere p1 h ∧ isInHemisphere p2 h ∧ isInHemisphere p3 h ∧ isInHemisphere p5 h) ∨
                      (isInHemisphere p1 h ∧ isInHemisphere p2 h ∧ isInHemisphere p4 h ∧ isInHemisphere p5 h) ∨
                      (isInHemisphere p1 h ∧ isInHemisphere p3 h ∧ isInHemisphere p4 h ∧ isInHemisphere p5 h) ∨
                      (isInHemisphere p2 h ∧ isInHemisphere p3 h ∧ isInHemisphere p4 h ∧ isInHemisphere p5 h) :=
sorry

end circle_theorem_sphere_theorem_l2556_255651


namespace brown_shoes_count_l2556_255692

theorem brown_shoes_count (brown_shoes black_shoes : ℕ) : 
  black_shoes = 2 * brown_shoes →
  brown_shoes + black_shoes = 66 →
  brown_shoes = 22 := by
sorry

end brown_shoes_count_l2556_255692


namespace food_for_six_days_is_87_l2556_255615

/-- Represents the daily food consumption for Joy's foster dogs -/
def daily_food_consumption : ℚ :=
  -- Mom's food
  (1.5 * 3) +
  -- First two puppies
  (2 * (1/2 * 3)) +
  -- Next two puppies
  (2 * (3/4 * 2)) +
  -- Last puppy
  (1 * 4)

/-- The total amount of food needed for 6 days -/
def total_food_for_six_days : ℚ := daily_food_consumption * 6

/-- Theorem stating that the total food needed for 6 days is 87 cups -/
theorem food_for_six_days_is_87 : total_food_for_six_days = 87 := by sorry

end food_for_six_days_is_87_l2556_255615


namespace draw_jack_queen_king_of_hearts_probability_l2556_255670

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (jacks : Nat)
  (queens : Nat)
  (king_of_hearts : Nat)

/-- The probability of drawing a specific sequence of cards from a deck -/
def draw_probability (d : Deck) : ℚ :=
  (d.jacks : ℚ) / d.total_cards *
  (d.queens : ℚ) / (d.total_cards - 1) *
  (d.king_of_hearts : ℚ) / (d.total_cards - 2)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  ⟨52, 4, 4, 1⟩

theorem draw_jack_queen_king_of_hearts_probability :
  draw_probability standard_deck = 4 / 33150 := by
  sorry

end draw_jack_queen_king_of_hearts_probability_l2556_255670


namespace log_sum_and_product_implies_arithmetic_mean_l2556_255681

theorem log_sum_and_product_implies_arithmetic_mean (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : Real.log x / Real.log y + Real.log y / Real.log x = 10/3) 
  (h4 : x * y = 144) : 
  (x + y) / 2 = 13 * Real.sqrt 3 := by
  sorry

end log_sum_and_product_implies_arithmetic_mean_l2556_255681


namespace difference_even_prime_sums_l2556_255663

def sumFirstNEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

def sumFirstNPrimes (n : ℕ) : ℕ := sorry

theorem difference_even_prime_sums : 
  sumFirstNEvenNumbers 3005 - sumFirstNPrimes 3005 = 9039030 - sumFirstNPrimes 3005 := by
  sorry

end difference_even_prime_sums_l2556_255663


namespace smallest_integer_l2556_255696

theorem smallest_integer (m n x : ℕ) : 
  m = 72 →
  x > 0 →
  Nat.gcd m n = x + 8 →
  Nat.lcm m n = x * (x + 8) →
  n ≥ 8 ∧ (∃ (y : ℕ), y > 0 ∧ y + 8 ∣ 72 ∧ y < x → False) :=
by sorry

end smallest_integer_l2556_255696


namespace high_school_nine_games_l2556_255614

/-- The number of teams in the league -/
def num_teams : ℕ := 9

/-- The number of non-league games each team plays -/
def non_league_games : ℕ := 6

/-- Calculate the total number of games in a season -/
def total_games : ℕ := 
  (num_teams * (num_teams - 1) / 2) + (num_teams * non_league_games)

/-- Theorem stating that the total number of games is 90 -/
theorem high_school_nine_games : total_games = 90 := by
  sorry

end high_school_nine_games_l2556_255614


namespace factor_expression_l2556_255605

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := by
  sorry

end factor_expression_l2556_255605


namespace biased_die_expected_value_l2556_255662

/-- The expected value of winnings for a biased die roll -/
theorem biased_die_expected_value :
  let p_six : ℚ := 1/4  -- Probability of rolling a 6
  let p_other : ℚ := 3/4  -- Probability of rolling any other number
  let win_six : ℚ := 4  -- Winnings for rolling a 6
  let lose_other : ℚ := -1  -- Loss for rolling any other number
  p_six * win_six + p_other * lose_other = 1/4 := by
sorry

end biased_die_expected_value_l2556_255662


namespace lucys_journey_l2556_255613

theorem lucys_journey (total : ℚ) 
  (h1 : total / 4 + 25 + total / 6 = total) : total = 300 / 7 := by
  sorry

end lucys_journey_l2556_255613


namespace work_rate_problem_l2556_255661

theorem work_rate_problem (A B C : ℚ) 
  (h1 : A + B = 1/8)
  (h2 : B + C = 1/12)
  (h3 : A + B + C = 1/6) :
  A + C = 1/8 := by
  sorry

end work_rate_problem_l2556_255661


namespace expression_evaluation_l2556_255632

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (2 * x - 1 / x) * (y - 1 / y) = -2 * x^2 + y^2 := by
  sorry

end expression_evaluation_l2556_255632


namespace percentage_before_break_l2556_255639

/-- Given a total number of pages and the number of pages to read after a break,
    calculate the percentage of pages that must be read before the break. -/
theorem percentage_before_break (total_pages : ℕ) (pages_after_break : ℕ) 
    (h1 : total_pages = 30) (h2 : pages_after_break = 9) : 
    (((total_pages - pages_after_break : ℚ) / total_pages) * 100 = 70) := by
  sorry

end percentage_before_break_l2556_255639


namespace second_investment_value_l2556_255668

theorem second_investment_value (x : ℝ) : 
  (0.07 * 500 + 0.09 * x = 0.085 * (500 + x)) → x = 1500 := by
  sorry

end second_investment_value_l2556_255668


namespace quadratic_root_implies_u_value_l2556_255622

theorem quadratic_root_implies_u_value (u : ℝ) :
  ((-15 - Real.sqrt 145) / 6 : ℝ) ∈ {x : ℝ | 3 * x^2 + 15 * x + u = 0} →
  u = 20/3 := by
sorry

end quadratic_root_implies_u_value_l2556_255622


namespace twenty_four_shots_hit_ship_l2556_255610

/-- Represents a point on the grid -/
structure Point where
  x : Fin 10
  y : Fin 10

/-- Represents a 1x4 ship on the grid -/
structure Ship where
  start : Point
  horizontal : Bool

/-- The set of 24 shots -/
def shots : Set Point := sorry

/-- Predicate to check if a ship overlaps with a point -/
def shipOverlapsPoint (s : Ship) (p : Point) : Prop := sorry

theorem twenty_four_shots_hit_ship :
  ∀ s : Ship, ∃ p ∈ shots, shipOverlapsPoint s p := by sorry

end twenty_four_shots_hit_ship_l2556_255610


namespace binary_1011001_equals_quaternary_1121_l2556_255688

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (λ x acc => 2 * acc + if x then 1 else 0) 0

def decimal_to_quaternary (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List (Fin 4) :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

theorem binary_1011001_equals_quaternary_1121 :
  decimal_to_quaternary (binary_to_decimal [true, false, true, true, false, false, true]) = [1, 1, 2, 1] := by
  sorry

end binary_1011001_equals_quaternary_1121_l2556_255688


namespace function_property_l2556_255679

/-- A function satisfying the given condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

/-- The main theorem -/
theorem function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x := by
  sorry

end function_property_l2556_255679


namespace sandy_fish_count_l2556_255601

def final_fish_count (initial : ℕ) (bought : ℕ) (given_away : ℕ) (babies : ℕ) : ℕ :=
  initial + bought - given_away + babies

theorem sandy_fish_count :
  final_fish_count 26 6 10 15 = 37 := by
  sorry

end sandy_fish_count_l2556_255601


namespace blanch_lunch_slices_l2556_255628

/-- The number of pizza slices Blanch ate during lunch -/
def lunch_slices (initial : ℕ) (breakfast : ℕ) (snack : ℕ) (dinner : ℕ) (remaining : ℕ) : ℕ :=
  initial - breakfast - snack - dinner - remaining

/-- Theorem stating that Blanch ate 2 slices during lunch -/
theorem blanch_lunch_slices :
  lunch_slices 15 4 2 5 2 = 2 := by sorry

end blanch_lunch_slices_l2556_255628


namespace daniels_age_l2556_255665

theorem daniels_age (uncle_bob_age : ℕ) (elizabeth_age : ℕ) (daniel_age : ℕ) :
  uncle_bob_age = 60 →
  elizabeth_age = (2 * uncle_bob_age) / 3 →
  daniel_age = elizabeth_age - 10 →
  daniel_age = 30 :=
by
  sorry

end daniels_age_l2556_255665


namespace largest_c_for_three_in_range_l2556_255626

/-- The function f(x) = x^2 - 7x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 7*x + c

/-- 3 is in the range of f -/
def three_in_range (c : ℝ) : Prop := ∃ x, f c x = 3

/-- The largest value of c such that 3 is in the range of f(x) = x^2 - 7x + c is 61/4 -/
theorem largest_c_for_three_in_range :
  (∃ c, three_in_range c ∧ ∀ c', three_in_range c' → c' ≤ c) ∧
  (∀ c, three_in_range c → c ≤ 61/4) :=
sorry

end largest_c_for_three_in_range_l2556_255626


namespace geometric_sequence_product_l2556_255677

/-- Given a geometric sequence {a_n} where a_1 and a_10 are the roots of 2x^2 + 5x + 1 = 0,
    prove that a_4 * a_7 = 1/2 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  (2 * (a 1)^2 + 5 * (a 1) + 1 = 0) →       -- a_1 is a root
  (2 * (a 10)^2 + 5 * (a 10) + 1 = 0) →     -- a_10 is a root
  a 4 * a 7 = 1/2 := by
sorry

end geometric_sequence_product_l2556_255677


namespace tangent_line_min_slope_l2556_255664

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 3

-- Theorem statement
theorem tangent_line_min_slope :
  ∃ (a b : ℝ), 
    (∀ x : ℝ, f' x ≥ f' a) ∧ 
    (∀ x : ℝ, f x = f a + f' a * (x - a)) ∧ 
    (b = -3 * a) ∧
    (∀ x : ℝ, f x = f a + b * (x - a)) :=
sorry

end tangent_line_min_slope_l2556_255664


namespace inequality_holds_iff_m_in_range_l2556_255623

theorem inequality_holds_iff_m_in_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1 / (x + 1) + 4 / y = 1) :
  (∀ m : ℝ, x + y / 4 > m^2 - 5*m - 3) ↔ ∀ m : ℝ, -1 < m ∧ m < 6 :=
by sorry

end inequality_holds_iff_m_in_range_l2556_255623


namespace max_x5y_given_constraint_l2556_255652

theorem max_x5y_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * (x + 2 * y) = 9) :
  x^5 * y ≤ 54 ∧ ∃ x0 y0 : ℝ, x0 > 0 ∧ y0 > 0 ∧ x0 * (x0 + 2 * y0) = 9 ∧ x0^5 * y0 = 54 :=
sorry

end max_x5y_given_constraint_l2556_255652


namespace smallest_n_below_threshold_l2556_255619

/-- The number of boxes in the warehouse -/
def num_boxes : ℕ := 2023

/-- The probability of drawing a green marble on the nth draw -/
def Q (n : ℕ) : ℚ := 1 / (n * (2 * n + 1))

/-- The threshold probability -/
def threshold : ℚ := 1 / num_boxes

/-- 32 is the smallest positive integer n such that Q(n) < 1/2023 -/
theorem smallest_n_below_threshold : 
  (∀ k < 32, Q k ≥ threshold) ∧ Q 32 < threshold :=
sorry

end smallest_n_below_threshold_l2556_255619


namespace student_count_l2556_255690

theorem student_count (n : ℕ) (rank_top : ℕ) (rank_bottom : ℕ) 
  (h1 : rank_top = 30) 
  (h2 : rank_bottom = 30) : 
  n = 59 := by
  sorry

end student_count_l2556_255690


namespace same_color_probability_l2556_255675

def total_plates : ℕ := 11
def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def selected_plates : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates selected_plates : ℚ) / (Nat.choose total_plates selected_plates) = 4 / 33 := by
  sorry

end same_color_probability_l2556_255675


namespace fixed_fee_is_5_20_l2556_255602

/-- Represents a music streaming service with a fixed monthly fee and a per-song fee -/
structure StreamingService where
  fixedFee : ℝ
  perSongFee : ℝ

/-- Calculates the total bill for a given number of songs -/
def bill (s : StreamingService) (songs : ℕ) : ℝ :=
  s.fixedFee + s.perSongFee * songs

theorem fixed_fee_is_5_20 (s : StreamingService) :
  bill s 20 = 15.20 ∧ bill s 40 = 25.20 → s.fixedFee = 5.20 := by
  sorry

#check fixed_fee_is_5_20

end fixed_fee_is_5_20_l2556_255602


namespace delores_purchase_shortage_delores_specific_shortage_l2556_255645

/-- Calculates the amount Delores is short by after attempting to purchase a computer, printer, and table -/
theorem delores_purchase_shortage (initial_amount : ℝ) (computer_price : ℝ) (computer_discount : ℝ)
  (printer_price : ℝ) (printer_tax : ℝ) (table_price_euros : ℝ) (exchange_rate : ℝ) : ℝ :=
  let computer_cost := computer_price * (1 - computer_discount)
  let printer_cost := printer_price * (1 + printer_tax)
  let table_cost := table_price_euros * exchange_rate
  let total_cost := computer_cost + printer_cost + table_cost
  total_cost - initial_amount

/-- Proves that Delores is short by $605 given the specific conditions -/
theorem delores_specific_shortage : 
  delores_purchase_shortage 450 1000 0.3 100 0.15 200 1.2 = 605 := by
  sorry


end delores_purchase_shortage_delores_specific_shortage_l2556_255645


namespace unique_m_for_power_function_l2556_255676

/-- A function f is a power function if it has the form f(x) = ax^b for some constants a and b, where a ≠ 0 -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^b

/-- A function f is increasing on (0, +∞) if for all x₁, x₂ > 0, x₁ < x₂ implies f(x₁) < f(x₂) -/
def is_increasing_on_positive_reals (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂

/-- The main theorem -/
theorem unique_m_for_power_function :
  ∃! m : ℝ, 
    is_power_function (fun x ↦ (m^2 - m - 1) * x^m) ∧
    is_increasing_on_positive_reals (fun x ↦ (m^2 - m - 1) * x^m) ∧
    m = 2 := by
  sorry

end unique_m_for_power_function_l2556_255676


namespace multiples_of_three_is_closed_l2556_255660

def is_closed (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

def multiples_of_three : Set ℤ :=
  {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem multiples_of_three_is_closed :
  is_closed multiples_of_three :=
by
  sorry

end multiples_of_three_is_closed_l2556_255660


namespace impossible_relationships_l2556_255600

theorem impossible_relationships (a b : ℝ) (h : 1 / a = 1 / b) :
  ¬(0 < a ∧ a < b) ∧ ¬(b < a ∧ a < 0) := by
  sorry

end impossible_relationships_l2556_255600


namespace descent_time_l2556_255617

/-- Prove that the time to descend a hill is 2 hours given the specified conditions -/
theorem descent_time (climb_time : ℝ) (climb_speed : ℝ) (total_avg_speed : ℝ) :
  climb_time = 4 →
  climb_speed = 2.625 →
  total_avg_speed = 3.5 →
  ∃ (descent_time : ℝ),
    descent_time = 2 ∧
    (2 * climb_time * climb_speed) = (total_avg_speed * (climb_time + descent_time)) :=
by sorry

end descent_time_l2556_255617


namespace f_is_even_l2556_255693

-- Define the function
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by
  sorry

end f_is_even_l2556_255693


namespace min_distance_ellipse_to_Q_l2556_255606

/-- The ellipse with semi-major axis 4 and semi-minor axis 2 -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

/-- The point Q -/
def Q : ℝ × ℝ := (2, 0)

/-- The squared distance between two points -/
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem min_distance_ellipse_to_Q :
  ∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧
    (∀ (P' : ℝ × ℝ), ellipse P'.1 P'.2 →
      distance_squared P Q ≤ distance_squared P' Q) ∧
    distance_squared P Q = (2*Real.sqrt 6/3)^2 := by
  sorry

end min_distance_ellipse_to_Q_l2556_255606


namespace tire_price_proof_l2556_255607

/-- The regular price of one tire -/
def regular_price : ℝ := 79

/-- The sale price of the fourth tire -/
def fourth_tire_price : ℝ := 3

/-- The total cost of four tires -/
def total_cost : ℝ := 240

theorem tire_price_proof :
  3 * regular_price + fourth_tire_price = total_cost :=
by sorry

end tire_price_proof_l2556_255607


namespace bridge_length_specific_bridge_length_l2556_255657

/-- The length of a bridge that a train can cross, given the train's length, speed, and time to cross. -/
theorem bridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time_s : ℝ) : ℝ :=
let train_speed_ms := train_speed_kmh * (1000 / 3600)
let total_distance := train_speed_ms * crossing_time_s
total_distance - train_length

/-- Proof that a bridge is 227 meters long given specific conditions -/
theorem specific_bridge_length : 
  bridge_length 148 45 30 = 227 := by
sorry

end bridge_length_specific_bridge_length_l2556_255657


namespace final_replacement_weight_is_140_l2556_255629

/-- The weight of the final replacement person in a series of replacements --/
def final_replacement_weight (initial_people : ℕ) (initial_weight : ℝ) 
  (first_increase : ℝ) (second_decrease : ℝ) (third_increase : ℝ) : ℝ :=
  let first_replacement := initial_weight + initial_people * first_increase
  let second_replacement := first_replacement - initial_people * second_decrease
  second_replacement + initial_people * third_increase - 
    (second_replacement - initial_people * second_decrease)

/-- Theorem stating the weight of the final replacement person --/
theorem final_replacement_weight_is_140 :
  final_replacement_weight 10 70 4 2 5 = 140 := by
  sorry


end final_replacement_weight_is_140_l2556_255629


namespace find_a_and_b_l2556_255683

theorem find_a_and_b : ∃ a b : ℤ, 
  (a - b = 831) ∧ 
  (a = 21 * b + 11) ∧ 
  (a = 872) ∧ 
  (b = 41) := by
  sorry

end find_a_and_b_l2556_255683


namespace perfect_square_trinomial_iff_m_eq_7_or_neg_1_l2556_255604

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number k such that
    ax^2 + bx + c = (√a * x + k)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + k)^2

/-- The main theorem stating that m = 7 or m = -1 if and only if
    x^2 + 2(m-3)x + 16 is a perfect square trinomial -/
theorem perfect_square_trinomial_iff_m_eq_7_or_neg_1 :
  ∀ m : ℝ, (m = 7 ∨ m = -1) ↔ is_perfect_square_trinomial 1 (2*(m-3)) 16 :=
by sorry

end perfect_square_trinomial_iff_m_eq_7_or_neg_1_l2556_255604


namespace range_of_k_l2556_255698

-- Define the complex number z
variable (z : ℂ)

-- Define sets A and B
def A (m k : ℝ) : Set ℂ :=
  {z | z = (2*m - Real.log (k+1)/k / Real.log (Real.sqrt 2)) + (m + Real.log (k+1)/k / Real.log (Real.sqrt 2)) * Complex.I}

def B (m : ℝ) : Set ℂ :=
  {z | Complex.abs z ≤ 2*m - 1}

-- Define the theorem
theorem range_of_k (m : ℝ) :
  (∀ k : ℝ, (A m k) ∩ (B m) = ∅) ↔ 
  ((4 * Real.sqrt 2 + 1) / 31 < k ∧ k < Real.sqrt 2 + 1) :=
sorry

end range_of_k_l2556_255698


namespace probability_triangle_or_circle_l2556_255611

theorem probability_triangle_or_circle (total : ℕ) (triangles : ℕ) (circles : ℕ) (squares : ℕ)
  (h_total : total = 10)
  (h_triangles : triangles = 4)
  (h_circles : circles = 3)
  (h_squares : squares = 3)
  (h_sum : triangles + circles + squares = total) :
  (triangles + circles : ℚ) / total = 7 / 10 := by
sorry

end probability_triangle_or_circle_l2556_255611


namespace matrix_problem_l2556_255672

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -1; -4, 3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![2, -2; -4, 6]

theorem matrix_problem :
  (∃ X : Matrix (Fin 2) (Fin 2) ℚ, A * X = B) ∧
  A⁻¹ = !![3/2, 1/2; 2, 1] ∧
  A * !![1, 0; 0, 2] = B := by sorry

end matrix_problem_l2556_255672


namespace distinct_subsets_remain_distinct_after_removal_l2556_255633

universe u

theorem distinct_subsets_remain_distinct_after_removal 
  {α : Type u} [DecidableEq α] (A : Finset α) (n : ℕ) 
  (subsets : Fin n → Finset α)
  (h_subset : ∀ i, (subsets i) ⊆ A)
  (h_distinct : ∀ i j, i ≠ j → subsets i ≠ subsets j) :
  ∃ a ∈ A, ∀ i j, i ≠ j → 
    (subsets i).erase a ≠ (subsets j).erase a :=
sorry

end distinct_subsets_remain_distinct_after_removal_l2556_255633


namespace smallest_number_is_57_l2556_255691

theorem smallest_number_is_57 (a b c d : ℕ) 
  (sum_abc : a + b + c = 234)
  (sum_abd : a + b + d = 251)
  (sum_acd : a + c + d = 284)
  (sum_bcd : b + c + d = 299) :
  min a (min b (min c d)) = 57 := by
sorry

end smallest_number_is_57_l2556_255691


namespace max_y_over_x_on_circle_l2556_255634

theorem max_y_over_x_on_circle (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) : 
  ∃ (max : ℝ), (∀ (a b : ℝ), (a - 2)^2 + b^2 = 3 → b / a ≤ max) ∧ max = Real.sqrt 3 := by
sorry

end max_y_over_x_on_circle_l2556_255634


namespace decimal_23_to_binary_binary_to_decimal_23_l2556_255678

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_23_to_binary :
  toBinary 23 = [true, true, true, false, true] :=
sorry

theorem binary_to_decimal_23 :
  fromBinary [true, true, true, false, true] = 23 :=
sorry

end decimal_23_to_binary_binary_to_decimal_23_l2556_255678


namespace inequality_solution_and_a_range_l2556_255687

def f (x : ℝ) := |3*x + 2|

theorem inequality_solution_and_a_range :
  (∀ x : ℝ, f x < 6 - |x - 2| ↔ -3/2 < x ∧ x < 1) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → m + n = 4 →
    (∀ a : ℝ, a > 0 →
      (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) →
        0 < a ∧ a ≤ 1/3)) :=
by sorry

end inequality_solution_and_a_range_l2556_255687


namespace pizza_toppings_combinations_l2556_255608

-- Define the number of available toppings
def n : ℕ := 9

-- Define the number of toppings to choose
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove
theorem pizza_toppings_combinations :
  combination n k = 84 := by
  sorry

end pizza_toppings_combinations_l2556_255608


namespace quadratic_trinomial_from_complete_square_l2556_255625

/-- 
Given a quadratic trinomial p(x) = Ax² + Bx + C, if its complete square form 
is x⁴ - 6x³ + 7x² + ax + b, then p(x) = x² - 3x - 1 or p(x) = -x² + 3x + 1.
-/
theorem quadratic_trinomial_from_complete_square (A B C a b : ℝ) :
  (∀ x, A * x^2 + B * x + C = x^4 - 6*x^3 + 7*x^2 + a*x + b) →
  ((A = 1 ∧ B = -3 ∧ C = -1) ∨ (A = -1 ∧ B = 3 ∧ C = 1)) :=
by sorry

end quadratic_trinomial_from_complete_square_l2556_255625


namespace expression_evaluation_l2556_255655

theorem expression_evaluation :
  let x : ℚ := -2/5
  let y : ℚ := 2
  2 * (x^2 * y - 2 * x * y) - 3 * (x^2 * y - 3 * x * y) + x^2 * y = -4 := by
sorry

end expression_evaluation_l2556_255655


namespace cistern_fill_time_l2556_255684

/-- Represents the time taken to fill a cistern given the rates of three pipes -/
def fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating that given pipes with specific rates will fill the cistern in 7.5 hours -/
theorem cistern_fill_time :
  fill_time (1/10) (1/12) (-1/20) = 15/2 := by
  sorry

end cistern_fill_time_l2556_255684


namespace polynomial_sum_l2556_255685

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) : 
  (∃ (x : ℝ), f a b x = g c d x ∧ f a b x = -25 ∧ x = 50) →  -- f and g intersect at (50, -25)
  (∀ (x : ℝ), f a b x ≥ -25) →  -- minimum value of f is -25
  (∀ (x : ℝ), g c d x ≥ -25) →  -- minimum value of g is -25
  g c d (-a/2) = 0 →  -- vertex of f is root of g
  f a b (-c/2) = 0 →  -- vertex of g is root of f
  a ≠ c →  -- f and g are distinct
  a + c = -101 := by
sorry

end polynomial_sum_l2556_255685
