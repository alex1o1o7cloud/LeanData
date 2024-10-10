import Mathlib

namespace pieces_per_package_calculation_l3274_327450

/-- Given the number of gum packages, candy packages, and total pieces,
    calculate the number of pieces per package. -/
def pieces_per_package (gum_packages : ℕ) (candy_packages : ℕ) (total_pieces : ℕ) : ℚ :=
  total_pieces / (gum_packages + candy_packages)

/-- Theorem stating that with 28 gum packages, 14 candy packages, and 7 total pieces,
    the number of pieces per package is 1/6. -/
theorem pieces_per_package_calculation :
  pieces_per_package 28 14 7 = 1/6 := by
  sorry

#eval pieces_per_package 28 14 7

end pieces_per_package_calculation_l3274_327450


namespace solve_system_l3274_327410

theorem solve_system (x y : ℝ) (eq1 : x + y = 15) (eq2 : x - y = 5) : y = 5 := by
  sorry

end solve_system_l3274_327410


namespace first_group_weight_proof_l3274_327469

-- Define the number of girls in the second group
def second_group_count : ℕ := 8

-- Define the average weights
def first_group_avg : ℝ := 50.25
def second_group_avg : ℝ := 45.15
def total_avg : ℝ := 48.55

-- Define the theorem
theorem first_group_weight_proof :
  ∃ (first_group_count : ℕ),
    (first_group_count * first_group_avg + second_group_count * second_group_avg) / 
    (first_group_count + second_group_count) = total_avg →
    first_group_avg = 50.25 := by
  sorry


end first_group_weight_proof_l3274_327469


namespace alice_paid_48_percent_of_srp_l3274_327417

-- Define the suggested retail price (SRP)
def suggested_retail_price : ℝ := 100

-- Define the marked price (MP) as 80% of SRP
def marked_price : ℝ := 0.8 * suggested_retail_price

-- Define Alice's purchase price as 60% of MP
def alice_price : ℝ := 0.6 * marked_price

-- Theorem to prove
theorem alice_paid_48_percent_of_srp :
  alice_price / suggested_retail_price = 0.48 := by
  sorry

end alice_paid_48_percent_of_srp_l3274_327417


namespace product_cost_price_l3274_327479

theorem product_cost_price (original_price : ℝ) (cost_price : ℝ) : 
  (0.8 * original_price - cost_price = 120) →
  (0.6 * original_price - cost_price = -20) →
  cost_price = 440 := by
  sorry

end product_cost_price_l3274_327479


namespace tan_neg_390_degrees_l3274_327468

theorem tan_neg_390_degrees : Real.tan ((-390 : ℝ) * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_neg_390_degrees_l3274_327468


namespace min_value_sum_squares_l3274_327428

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + y + 2*z = 6) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), 2*a + b + 2*c = 6 → x^2 + y^2 + z^2 ≥ m ∧ a^2 + b^2 + c^2 ≥ m :=
sorry

end min_value_sum_squares_l3274_327428


namespace race_speed_ratio_l3274_327440

/-- Proves that the ratio of A's speed to B's speed is 2:1 in a race where A gives B a head start -/
theorem race_speed_ratio (race_length : ℝ) (head_start : ℝ) (speed_A : ℝ) (speed_B : ℝ) 
  (h1 : race_length = 142)
  (h2 : head_start = 71)
  (h3 : race_length / speed_A = (race_length - head_start) / speed_B) :
  speed_A / speed_B = 2 := by
  sorry

#check race_speed_ratio

end race_speed_ratio_l3274_327440


namespace clubsuit_ratio_l3274_327421

-- Define the ♣ operation
def clubsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem clubsuit_ratio : (clubsuit 3 5) / (clubsuit 5 3) = 5 / 3 := by
  sorry

end clubsuit_ratio_l3274_327421


namespace rice_mixture_cost_problem_l3274_327482

/-- The cost of the second variety of rice per kg -/
def second_variety_cost : ℝ := 12.50

/-- The cost of the first variety of rice per kg -/
def first_variety_cost : ℝ := 5

/-- The cost of the mixture per kg -/
def mixture_cost : ℝ := 7.50

/-- The ratio of the two varieties of rice -/
def rice_ratio : ℝ := 0.5

theorem rice_mixture_cost_problem :
  first_variety_cost * 1 + second_variety_cost * rice_ratio = mixture_cost * (1 + rice_ratio) :=
by sorry

end rice_mixture_cost_problem_l3274_327482


namespace johns_bill_total_l3274_327487

/-- Calculates the total amount due on a bill after applying late charges and annual interest. -/
def totalAmountDue (originalBill : ℝ) (lateChargeRate : ℝ) (numLateCharges : ℕ) (annualInterestRate : ℝ) : ℝ :=
  let afterLateCharges := originalBill * (1 + lateChargeRate) ^ numLateCharges
  afterLateCharges * (1 + annualInterestRate)

/-- Proves that the total amount due on John's bill is $557.13 after one year. -/
theorem johns_bill_total : 
  let originalBill : ℝ := 500
  let lateChargeRate : ℝ := 0.02
  let numLateCharges : ℕ := 3
  let annualInterestRate : ℝ := 0.05
  totalAmountDue originalBill lateChargeRate numLateCharges annualInterestRate = 557.13 := by
  sorry


end johns_bill_total_l3274_327487


namespace beats_played_example_l3274_327485

/-- Given a person who plays music at a certain rate for a specific duration each day over multiple days, calculate the total number of beats played. -/
def totalBeatsPlayed (beatsPerMinute : ℕ) (hoursPerDay : ℕ) (numberOfDays : ℕ) : ℕ :=
  beatsPerMinute * (hoursPerDay * 60) * numberOfDays

/-- Theorem stating that playing 200 beats per minute for 2 hours a day for 3 days results in 72,000 beats total. -/
theorem beats_played_example : totalBeatsPlayed 200 2 3 = 72000 := by
  sorry

end beats_played_example_l3274_327485


namespace julios_age_l3274_327477

/-- Proves that Julio's current age is 36 years old, given the conditions of the problem -/
theorem julios_age (james_age : ℕ) (future_years : ℕ) (julio_age : ℕ) : 
  james_age = 11 →
  future_years = 14 →
  julio_age + future_years = 2 * (james_age + future_years) →
  julio_age = 36 :=
by sorry

end julios_age_l3274_327477


namespace function_simplification_and_sum_l3274_327462

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 - 4*x - 12) / (x - 3)

theorem function_simplification_and_sum :
  ∃ (A B C D : ℝ),
    (∀ x : ℝ, x ≠ D → f x = A * x^2 + B * x + C) ∧
    (∀ x : ℝ, f x = A * x^2 + B * x + C ↔ x ≠ D) ∧
    A + B + C + D = 24 := by
  sorry

end function_simplification_and_sum_l3274_327462


namespace arithmetic_mean_expressions_l3274_327404

theorem arithmetic_mean_expressions (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x + a) / y + (y - b) / x) / 2 = (x^2 + a*x + y^2 - b*y) / (2*x*y) := by
  sorry

end arithmetic_mean_expressions_l3274_327404


namespace function_inequality_l3274_327437

theorem function_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : Real.exp (a + 1) = a + 4) (h2 : Real.log (b + 3) = b) :
  let f := fun x => Real.exp x + (a - b) * x
  f (2/3) < f 0 ∧ f 0 < f 2 := by
  sorry

end function_inequality_l3274_327437


namespace athul_rowing_time_l3274_327415

theorem athul_rowing_time (upstream_distance : ℝ) (downstream_distance : ℝ) (stream_speed : ℝ) :
  upstream_distance = 16 →
  downstream_distance = 24 →
  stream_speed = 1 →
  ∃ (rowing_speed : ℝ),
    rowing_speed > stream_speed ∧
    (upstream_distance / (rowing_speed - stream_speed) = downstream_distance / (rowing_speed + stream_speed)) ∧
    (upstream_distance / (rowing_speed - stream_speed) = 4) :=
by
  sorry

end athul_rowing_time_l3274_327415


namespace cryptarithm_solution_l3274_327461

theorem cryptarithm_solution :
  ∃! (C H U K T R I G N S : ℕ),
    C < 10 ∧ H < 10 ∧ U < 10 ∧ K < 10 ∧ T < 10 ∧ R < 10 ∧ I < 10 ∧ G < 10 ∧ N < 10 ∧ S < 10 ∧
    T ≠ 0 ∧
    C ≠ H ∧ C ≠ U ∧ C ≠ K ∧ C ≠ T ∧ C ≠ R ∧ C ≠ I ∧ C ≠ G ∧ C ≠ N ∧ C ≠ S ∧
    H ≠ U ∧ H ≠ K ∧ H ≠ T ∧ H ≠ R ∧ H ≠ I ∧ H ≠ G ∧ H ≠ N ∧ H ≠ S ∧
    U ≠ K ∧ U ≠ T ∧ U ≠ R ∧ U ≠ I ∧ U ≠ G ∧ U ≠ N ∧ U ≠ S ∧
    K ≠ T ∧ K ≠ R ∧ K ≠ I ∧ K ≠ G ∧ K ≠ N ∧ K ≠ S ∧
    T ≠ R ∧ T ≠ I ∧ T ≠ G ∧ T ≠ N ∧ T ≠ S ∧
    R ≠ I ∧ R ≠ G ∧ R ≠ N ∧ R ≠ S ∧
    I ≠ G ∧ I ≠ N ∧ I ≠ S ∧
    G ≠ N ∧ G ≠ S ∧
    N ≠ S ∧
    100000*C + 10000*H + 1000*U + 100*C + 10*K +
    100000*T + 10000*R + 1000*I + 100*G + 10*G +
    100000*T + 10000*U + 1000*R + 100*N + 10*S =
    100000*T + 10000*R + 1000*I + 100*C + 10*K + S ∧
    C = 9 ∧ H = 3 ∧ U = 5 ∧ K = 4 ∧ T = 1 ∧ R = 2 ∧ I = 0 ∧ G = 6 ∧ N = 8 ∧ S = 7 := by
  sorry

end cryptarithm_solution_l3274_327461


namespace quadratic_roots_range_l3274_327424

theorem quadratic_roots_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11*x + (30 + k) = 0 → x > 5) → 
  0 < k ∧ k ≤ 1/4 := by
sorry

end quadratic_roots_range_l3274_327424


namespace solution_and_rationality_l3274_327497

theorem solution_and_rationality 
  (x y : ℝ) 
  (h : Real.sqrt (8 * x - y^2) + |y^2 - 16| = 0) : 
  (x = 2 ∧ (y = 4 ∨ y = -4)) ∧ 
  ((y = 4 → ∃ (q : ℚ), Real.sqrt (y + 12) = ↑q) ∧ 
   (y = -4 → ∀ (q : ℚ), Real.sqrt (y + 12) ≠ ↑q)) := by
  sorry

end solution_and_rationality_l3274_327497


namespace petes_bottle_return_l3274_327489

/-- Represents the number of bottles Pete needs to return to the store -/
def bottles_to_return (total_owed : ℚ) (cash_in_wallet : ℚ) (cash_in_pockets : ℚ) (bottle_return_rate : ℚ) : ℕ :=
  sorry

/-- The theorem stating the number of bottles Pete needs to return -/
theorem petes_bottle_return : 
  bottles_to_return 90 40 40 (1/2) = 20 := by sorry

end petes_bottle_return_l3274_327489


namespace two_digit_number_square_equals_cube_of_digit_sum_l3274_327488

theorem two_digit_number_square_equals_cube_of_digit_sum :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧
  (∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧ n = 10 * a + b) ∧
  n^2 = (n / 10 + n % 10)^3 :=
by sorry

end two_digit_number_square_equals_cube_of_digit_sum_l3274_327488


namespace intersection_complement_equality_l3274_327400

-- Define the sets U, A, and B
def U : Set ℝ := {x | x ≤ -1 ∨ x ≥ 0}
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (U \ B) = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end intersection_complement_equality_l3274_327400


namespace total_food_items_is_149_l3274_327457

/-- Represents the eating habits of a person -/
structure EatingHabits where
  croissants : ℕ
  cakes : ℕ
  pizzas : ℕ

/-- Calculates the total food items consumed by a person -/
def totalFoodItems (habits : EatingHabits) : ℕ :=
  habits.croissants + habits.cakes + habits.pizzas

/-- The eating habits of Jorge -/
def jorge : EatingHabits :=
  { croissants := 7, cakes := 18, pizzas := 30 }

/-- The eating habits of Giuliana -/
def giuliana : EatingHabits :=
  { croissants := 5, cakes := 14, pizzas := 25 }

/-- The eating habits of Matteo -/
def matteo : EatingHabits :=
  { croissants := 6, cakes := 16, pizzas := 28 }

/-- Theorem stating that the total food items consumed by Jorge, Giuliana, and Matteo is 149 -/
theorem total_food_items_is_149 :
  totalFoodItems jorge + totalFoodItems giuliana + totalFoodItems matteo = 149 := by
  sorry

end total_food_items_is_149_l3274_327457


namespace y_intercept_of_perpendicular_line_l3274_327448

-- Define line l
def line_l (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define perpendicularity of two lines given their slopes
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Define a point on a line given its slope and a point it passes through
def point_on_line (m x₀ y₀ y : ℝ) (x : ℝ) : Prop := y - y₀ = m * (x - x₀)

-- Theorem statement
theorem y_intercept_of_perpendicular_line :
  ∃ (m : ℝ), 
    (∀ x y, line_l x y → y = (1/2) * x + (1/2)) →
    perpendicular (1/2) m →
    point_on_line m (-1) 0 0 0 →
    ∃ y, point_on_line m 0 y 0 0 ∧ y = -2 := by
  sorry

end y_intercept_of_perpendicular_line_l3274_327448


namespace inverse_functions_l3274_327456

-- Define the types of functions
def LinearDecreasing : Type := ℝ → ℝ
def PiecewiseConstant : Type := ℝ → ℝ
def VerticalLine : Type := ℝ → ℝ
def Semicircle : Type := ℝ → ℝ
def ModifiedPolynomial : Type := ℝ → ℝ

-- Define the property of having an inverse
def HasInverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x

-- State the theorem
theorem inverse_functions 
  (F : LinearDecreasing) 
  (G : PiecewiseConstant) 
  (H : VerticalLine) 
  (I : Semicircle) 
  (J : ModifiedPolynomial) : 
  HasInverse F ∧ HasInverse G ∧ ¬HasInverse H ∧ ¬HasInverse I ∧ ¬HasInverse J := by
  sorry

end inverse_functions_l3274_327456


namespace smallest_perfect_square_factor_l3274_327411

def y : ℕ := 2^5 * 3^2 * 4^6 * 5^6 * 7^8 * 8^9 * 9^10

theorem smallest_perfect_square_factor (k : ℕ) : 
  (k > 0 ∧ ∃ m : ℕ, k * y = m^2) → k ≥ 100 :=
sorry

end smallest_perfect_square_factor_l3274_327411


namespace sqrt_x_plus_reciprocal_l3274_327465

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
sorry

end sqrt_x_plus_reciprocal_l3274_327465


namespace path_area_and_cost_l3274_327449

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per square meter -/
def construction_cost (path_area cost_per_sqm : ℝ) : ℝ :=
  path_area * cost_per_sqm

theorem path_area_and_cost (field_length field_width path_width cost_per_sqm : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 3.5)
  (h4 : cost_per_sqm = 2) :
  path_area field_length field_width path_width = 959 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_sqm = 1918 := by
  sorry

#eval path_area 75 55 3.5
#eval construction_cost (path_area 75 55 3.5) 2

end path_area_and_cost_l3274_327449


namespace animal_population_canada_animal_population_l3274_327423

/-- The combined population of moose, beavers, caribou, wolves, grizzly bears, and mountain lions in Canada, given the specified ratios and human population. -/
theorem animal_population (human_population : ℝ) : ℝ :=
  let beaver_population := human_population / 19
  let moose_population := beaver_population / 2
  let caribou_population := 3/2 * moose_population
  let wolf_population := 4 * caribou_population
  let grizzly_population := wolf_population / 3
  let mountain_lion_population := grizzly_population / 2
  moose_population + beaver_population + caribou_population + wolf_population + grizzly_population + mountain_lion_population

/-- Theorem stating that the combined animal population in Canada is 13.5 million, given a human population of 38 million. -/
theorem canada_animal_population :
  animal_population 38 = 13.5 := by sorry

end animal_population_canada_animal_population_l3274_327423


namespace smallest_n_for_real_root_l3274_327454

/-- A polynomial with coefficients in [100, 101] -/
def PolynomialInRange (P : Polynomial ℝ) : Prop :=
  ∀ i, (100 : ℝ) ≤ P.coeff i ∧ P.coeff i ≤ 101

/-- The existence of a polynomial with a real root -/
def ExistsPolynomialWithRealRoot (n : ℕ) : Prop :=
  ∃ (P : Polynomial ℝ), PolynomialInRange P ∧ P.degree = 2*n ∧ ∃ x : ℝ, P.eval x = 0

/-- The main theorem stating that 100 is the smallest n for which a polynomial
    with coefficients in [100, 101] can have a real root -/
theorem smallest_n_for_real_root :
  (ExistsPolynomialWithRealRoot 100) ∧
  (∀ m : ℕ, m < 100 → ¬(ExistsPolynomialWithRealRoot m)) := by
  sorry

end smallest_n_for_real_root_l3274_327454


namespace min_sum_p_q_l3274_327494

theorem min_sum_p_q (p q : ℕ) : 
  p > 1 → q > 1 → 17 * (p + 1) = 28 * (q + 1) → 
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 28 * (q' + 1) → 
  p + q ≤ p' + q' → p + q = 135 := by
sorry

end min_sum_p_q_l3274_327494


namespace absolute_value_inequality_solution_set_l3274_327474

theorem absolute_value_inequality_solution_set : 
  {x : ℝ | |x| > -1} = Set.univ :=
sorry

end absolute_value_inequality_solution_set_l3274_327474


namespace hyperbola_conjugate_axis_length_l3274_327496

/-- Represents a hyperbola with equation x^2/5 - y^2/b^2 = 1 -/
structure Hyperbola where
  b : ℝ
  eq : ∀ x y : ℝ, x^2/5 - y^2/b^2 = 1

/-- The distance from the focus to the asymptote of the hyperbola -/
def focus_to_asymptote_distance (h : Hyperbola) : ℝ := 2

/-- The length of the conjugate axis of the hyperbola -/
def conjugate_axis_length (h : Hyperbola) : ℝ := 2 * h.b

theorem hyperbola_conjugate_axis_length (h : Hyperbola) :
  focus_to_asymptote_distance h = 2 →
  conjugate_axis_length h = 4 := by
  sorry

end hyperbola_conjugate_axis_length_l3274_327496


namespace problem_solution_l3274_327460

theorem problem_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ + 64*x₈ = 2)
  (eq2 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ + 100*x₈ = 24)
  (eq3 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ + 121*x₈ = 246)
  (eq4 : 25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ + 144*x₈ = 1234) :
  36*x₁ + 49*x₂ + 64*x₃ + 81*x₄ + 100*x₅ + 121*x₆ + 144*x₇ + 169*x₈ = 1594 := by
  sorry


end problem_solution_l3274_327460


namespace single_transmission_prob_triple_transmission_better_for_zero_l3274_327480

/-- Represents a binary communication channel with error probabilities α and β -/
structure BinaryChannel where
  α : ℝ
  β : ℝ
  α_pos : 0 < α
  α_lt_one : α < 1
  β_pos : 0 < β
  β_lt_one : β < 1

/-- Probability of receiving 1,0,1 when sending 1,0,1 in single transmission -/
def singleTransmissionProb (c : BinaryChannel) : ℝ :=
  (1 - c.α) * (1 - c.β)^2

/-- Probability of decoding 0 when sending 0 in single transmission -/
def singleTransmission0Prob (c : BinaryChannel) : ℝ :=
  1 - c.α

/-- Probability of decoding 0 when sending 0 in triple transmission -/
def tripleTransmission0Prob (c : BinaryChannel) : ℝ :=
  (1 - c.α)^3 + 3 * c.α * (1 - c.α)^2

theorem single_transmission_prob (c : BinaryChannel) :
  singleTransmissionProb c = (1 - c.α) * (1 - c.β)^2 := by sorry

theorem triple_transmission_better_for_zero (c : BinaryChannel) (h : c.α < 0.5) :
  singleTransmission0Prob c < tripleTransmission0Prob c := by sorry

end single_transmission_prob_triple_transmission_better_for_zero_l3274_327480


namespace line_tangent_to_circle_l3274_327464

/-- A circle with a diameter of 10 units -/
def Circle := {p : ℝ × ℝ | (p.1 ^ 2 + p.2 ^ 2) ≤ 25}

/-- A line at distance d from the origin -/
def Line (d : ℝ) := {p : ℝ × ℝ | p.2 = d}

/-- The line is tangent to the circle if and only if the distance is 5 -/
theorem line_tangent_to_circle (d : ℝ) : 
  (∃ (p : ℝ × ℝ), p ∈ Circle ∩ Line d ∧ 
    ∀ (q : ℝ × ℝ), q ∈ Circle ∩ Line d → q = p) ↔ 
  d = 5 :=
sorry

end line_tangent_to_circle_l3274_327464


namespace store_transaction_result_l3274_327444

/-- Represents the result of a store's transaction -/
inductive TransactionResult
  | BreakEven
  | Profit (amount : ℝ)
  | Loss (amount : ℝ)

/-- Calculates the result of a store's transaction given the selling price and profit/loss percentages -/
def calculateTransactionResult (sellingPrice : ℝ) (profit1 : ℝ) (loss2 : ℝ) : TransactionResult :=
  sorry

theorem store_transaction_result :
  let sellingPrice : ℝ := 80
  let profit1 : ℝ := 60
  let loss2 : ℝ := 20
  calculateTransactionResult sellingPrice profit1 loss2 = TransactionResult.Profit 10 :=
sorry

end store_transaction_result_l3274_327444


namespace IMO_2001_max_sum_l3274_327429

theorem IMO_2001_max_sum : 
  ∀ I M O : ℕ+,
  I ≠ M → I ≠ O → M ≠ O →
  I * M * O = 2001 →
  I + M + O ≤ 671 :=
by
  sorry

end IMO_2001_max_sum_l3274_327429


namespace chapters_per_book_l3274_327483

theorem chapters_per_book (total_books : ℕ) (total_chapters : ℕ) (h1 : total_books = 4) (h2 : total_chapters = 68) :
  total_chapters / total_books = 17 := by
  sorry

end chapters_per_book_l3274_327483


namespace bus_driver_compensation_l3274_327486

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_percentage : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 14)
  (h2 : regular_hours = 40)
  (h3 : overtime_percentage = 0.75)
  (h4 : total_hours = 57.88) :
  ∃ (total_compensation : ℝ), 
    abs (total_compensation - 998.06) < 0.01 ∧
    total_compensation = 
      regular_rate * regular_hours + 
      (regular_rate * (1 + overtime_percentage)) * (total_hours - regular_hours) :=
by sorry


end bus_driver_compensation_l3274_327486


namespace quadratic_solution_l3274_327455

theorem quadratic_solution (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 11 * x - 20 = 0) : x = 4/3 := by
  sorry

end quadratic_solution_l3274_327455


namespace cone_volume_l3274_327484

/-- The volume of a cone with lateral surface area 2√3π and central angle √3π is π. -/
theorem cone_volume (r l : ℝ) (h_angle : 2 * π * r / l = Real.sqrt 3 * π)
  (h_area : π * r * l = 2 * Real.sqrt 3 * π) : 
  (1/3) * π * r^2 * Real.sqrt (l^2 - r^2) = π :=
sorry

end cone_volume_l3274_327484


namespace liquid_rise_ratio_in_cones_l3274_327442

theorem liquid_rise_ratio_in_cones (r₁ r₂ r_marble : ℝ) 
  (h₁ h₂ : ℝ) (V : ℝ) :
  r₁ = 4 →
  r₂ = 8 →
  r_marble = 2 →
  V = (1/3) * π * r₁^2 * h₁ →
  V = (1/3) * π * r₂^2 * h₂ →
  let V_marble := (4/3) * π * r_marble^3
  let h₁' := h₁ + V_marble / ((1/3) * π * r₁^2)
  let h₂' := h₂ + V_marble / ((1/3) * π * r₂^2)
  (h₁' - h₁) / (h₂' - h₂) = 4 :=
by sorry

#check liquid_rise_ratio_in_cones

end liquid_rise_ratio_in_cones_l3274_327442


namespace described_method_is_analogical_thinking_l3274_327478

/-- A learning method in mathematics -/
structure LearningMethod where
  compare_objects : Bool
  find_similarities : Bool
  deduce_similar_properties : Bool

/-- Analogical thinking in mathematics -/
def analogical_thinking : LearningMethod :=
  { compare_objects := true,
    find_similarities := true,
    deduce_similar_properties := true }

/-- The described learning method -/
def described_method : LearningMethod :=
  { compare_objects := true,
    find_similarities := true,
    deduce_similar_properties := true }

/-- Theorem stating that the described learning method is equivalent to analogical thinking -/
theorem described_method_is_analogical_thinking : described_method = analogical_thinking :=
  sorry

end described_method_is_analogical_thinking_l3274_327478


namespace cosine_sine_sum_equals_half_l3274_327446

theorem cosine_sine_sum_equals_half : 
  Real.cos (36 * π / 180) * Real.cos (96 * π / 180) + 
  Real.sin (36 * π / 180) * Real.sin (84 * π / 180) = 1 / 2 := by
  sorry

end cosine_sine_sum_equals_half_l3274_327446


namespace boosters_club_average_sales_l3274_327447

/-- The average monthly sales for the Boosters Club candy sales --/
theorem boosters_club_average_sales :
  let sales : List ℕ := [90, 50, 70, 110, 80]
  let total_sales : ℕ := sales.sum
  let num_months : ℕ := sales.length
  (total_sales : ℚ) / num_months = 80 := by sorry

end boosters_club_average_sales_l3274_327447


namespace geometric_sequence_third_term_l3274_327473

/-- A geometric sequence with a negative common ratio -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q < 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_first : a 1 = 2)
  (h_relation : a 3 - 4 = a 2) :
  a 3 = 2 := by
  sorry


end geometric_sequence_third_term_l3274_327473


namespace peanut_butter_jars_l3274_327427

/-- Given 2032 ounces of peanut butter distributed equally among jars of 16, 28, 40, and 52 ounces,
    the total number of jars is 60. -/
theorem peanut_butter_jars :
  let total_ounces : ℕ := 2032
  let jar_sizes : List ℕ := [16, 28, 40, 52]
  let num_sizes : ℕ := jar_sizes.length
  ∃ (x : ℕ),
    (x * (jar_sizes.sum)) = total_ounces ∧
    (num_sizes * x) = 60
  := by sorry

end peanut_butter_jars_l3274_327427


namespace exists_valid_configuration_l3274_327413

/-- A configuration of 9 numbers placed in circles -/
def Configuration := Fin 9 → Nat

/-- The 6 lines connecting the circles -/
def Lines := Fin 6 → Fin 3 → Fin 9

/-- Check if a configuration is valid -/
def is_valid_configuration (config : Configuration) (lines : Lines) : Prop :=
  (∀ i : Fin 9, config i ∈ Finset.range 10 \ {0}) ∧  -- Numbers are from 1 to 9
  (∃ i : Fin 9, config i = 6) ∧                      -- 6 is included
  (∀ i j : Fin 9, i ≠ j → config i ≠ config j) ∧     -- All numbers are different
  (∀ l : Fin 6, (config (lines l 0) + config (lines l 1) + config (lines l 2) = 23))  -- Sum on each line is 23

theorem exists_valid_configuration (lines : Lines) : 
  ∃ (config : Configuration), is_valid_configuration config lines :=
sorry

end exists_valid_configuration_l3274_327413


namespace reading_time_difference_l3274_327471

theorem reading_time_difference 
  (xanthia_speed : ℝ) 
  (molly_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : xanthia_speed = 120) 
  (h2 : molly_speed = 60) 
  (h3 : book_pages = 300) : 
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 150 := by
  sorry

end reading_time_difference_l3274_327471


namespace orange_book_pages_l3274_327435

/-- Proves that the number of pages in each orange book is 510, given the specified conditions --/
theorem orange_book_pages : ℕ → Prop :=
  fun (x : ℕ) =>
    let purple_pages_per_book : ℕ := 230
    let purple_books_read : ℕ := 5
    let orange_books_read : ℕ := 4
    let extra_orange_pages : ℕ := 890
    (purple_pages_per_book * purple_books_read + extra_orange_pages = orange_books_read * x) →
    x = 510

/-- The proof of the theorem --/
lemma prove_orange_book_pages : orange_book_pages 510 := by
  sorry

end orange_book_pages_l3274_327435


namespace inequality_one_inequality_two_l3274_327408

-- First inequality
theorem inequality_one (x : ℝ) : 
  (|1 - (2*x - 1)/3| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 5) :=
sorry

-- Second inequality
theorem inequality_two (x : ℝ) :
  ((2 - x)*(x + 3) < 2 - x) ↔ (x > 2 ∨ x < -2) :=
sorry

end inequality_one_inequality_two_l3274_327408


namespace race_completion_time_l3274_327451

/-- Given a 1000-meter race where runner A beats runner B by either 60 meters or 10 seconds,
    this theorem proves that runner A completes the race in 156.67 seconds. -/
theorem race_completion_time :
  ∀ (speed_A speed_B : ℝ),
  speed_A > 0 ∧ speed_B > 0 →
  1000 / speed_A = 940 / speed_B →
  1000 / speed_A = (1000 / speed_B) - 10 →
  1000 / speed_A = 156.67 :=
by sorry

end race_completion_time_l3274_327451


namespace ferris_wheel_cost_l3274_327443

def rollercoaster_rides : ℕ := 3
def catapult_rides : ℕ := 2
def ferris_wheel_rides : ℕ := 1
def rollercoaster_cost : ℕ := 4
def catapult_cost : ℕ := 4
def total_tickets : ℕ := 21

theorem ferris_wheel_cost :
  total_tickets - (rollercoaster_rides * rollercoaster_cost + catapult_rides * catapult_cost) = ferris_wheel_rides := by
  sorry

end ferris_wheel_cost_l3274_327443


namespace equal_digit_probability_l3274_327420

def num_dice : ℕ := 6
def sides_per_die : ℕ := 16
def one_digit_prob : ℚ := 9 / 16
def two_digit_prob : ℚ := 7 / 16

theorem equal_digit_probability : 
  (num_dice.choose (num_dice / 2)) * (one_digit_prob ^ (num_dice / 2)) * (two_digit_prob ^ (num_dice / 2)) = 3115125 / 10485760 := by
  sorry

end equal_digit_probability_l3274_327420


namespace inequality_solution_l3274_327406

open Set Real

def inequality_holds (x a : ℝ) : Prop :=
  (a + 2) * x - (1 + 2 * a) * (x^2)^(1/3) - 6 * x^(1/3) + a^2 + 4 * a - 5 > 0

theorem inequality_solution :
  ∀ x : ℝ, (∃ a ∈ Icc (-2) 1, inequality_holds x a) ↔ 
  x ∈ Iio (-1) ∪ Ioo (-1) 0 ∪ Ioi 8 :=
sorry

end inequality_solution_l3274_327406


namespace solve_quadratic_equation_l3274_327433

theorem solve_quadratic_equation (k p : ℝ) (hk : k ≠ 0) (hp : p ≠ 0) :
  let y : ℝ := -(p + k^2) / (2*k)
  (y - 2*k)^2 - (y - 3*k)^2 = 4*k^2 - p := by
sorry

end solve_quadratic_equation_l3274_327433


namespace distance_to_origin_l3274_327441

/-- The distance from point (5, -12) to the origin in the Cartesian coordinate system is 13. -/
theorem distance_to_origin : Real.sqrt (5^2 + (-12)^2) = 13 := by
  sorry

end distance_to_origin_l3274_327441


namespace andrews_age_l3274_327405

theorem andrews_age :
  ∀ (a g : ℝ),
  g = 15 * a →
  g - a = 55 →
  a = 55 / 14 :=
by
  sorry

end andrews_age_l3274_327405


namespace prime_with_integer_roots_l3274_327432

theorem prime_with_integer_roots (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ x y : ℤ, x^2 + p*x - 204*p = 0 ∧ y^2 + p*y - 204*p = 0) → p = 17 := by
  sorry

end prime_with_integer_roots_l3274_327432


namespace probability_white_ball_l3274_327475

/-- The probability of drawing a white ball from a bag with black and white balls -/
theorem probability_white_ball (black_balls white_balls : ℕ) : 
  black_balls = 6 → white_balls = 5 → 
  (white_balls : ℚ) / (black_balls + white_balls : ℚ) = 5 / 11 :=
by
  sorry

#check probability_white_ball

end probability_white_ball_l3274_327475


namespace quadratic_sum_l3274_327466

/-- Given a quadratic function f(x) = -3x^2 - 27x + 81, prove that when 
    rewritten in the form a(x+b)^2 + c, the sum a + b + c equals 143.25 -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = -3*x^2 - 27*x + 81) →
  (∀ x, f x = a*(x+b)^2 + c) →
  a + b + c = 143.25 := by
  sorry

end quadratic_sum_l3274_327466


namespace rectangle_y_value_l3274_327414

theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  let vertices : List (ℝ × ℝ) := [(-2, y), (10, y), (-2, -1), (10, -1)]
  let length : ℝ := 10 - (-2)
  let height : ℝ := y - (-1)
  let area : ℝ := length * height
  area = 108 → y = 8 := by sorry

end rectangle_y_value_l3274_327414


namespace am_gm_inequality_l3274_327458

theorem am_gm_inequality (a b : ℝ) (h : a * b > 0) : a / b + b / a ≥ 2 := by
  sorry

end am_gm_inequality_l3274_327458


namespace array_exists_iff_even_l3274_327422

/-- A type representing the possible entries in the array -/
inductive Entry
  | neg : Entry
  | zero : Entry
  | pos : Entry

/-- Definition of a valid array -/
def ValidArray (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) : Prop :=
  ∀ (i j : Fin n), arr i j ∈ [Entry.neg, Entry.zero, Entry.pos]

/-- Definition of row sum -/
def RowSum (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) (i : Fin n) : ℤ :=
  (Finset.univ.sum fun j => match arr i j with
    | Entry.neg => -1
    | Entry.zero => 0
    | Entry.pos => 1)

/-- Definition of column sum -/
def ColSum (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) (j : Fin n) : ℤ :=
  (Finset.univ.sum fun i => match arr i j with
    | Entry.neg => -1
    | Entry.zero => 0
    | Entry.pos => 1)

/-- All sums are different -/
def AllSumsDifferent (n : ℕ) (arr : Matrix (Fin n) (Fin n) Entry) : Prop :=
  ∀ (i j i' j' : Fin n), 
    (RowSum n arr i = RowSum n arr i' → i = i') ∧
    (ColSum n arr j = ColSum n arr j' → j = j') ∧
    (RowSum n arr i ≠ ColSum n arr j)

/-- Main theorem: The array with described properties exists if and only if n is even -/
theorem array_exists_iff_even (n : ℕ) :
  (∃ (arr : Matrix (Fin n) (Fin n) Entry), 
    ValidArray n arr ∧ AllSumsDifferent n arr) ↔ Even n :=
sorry

end array_exists_iff_even_l3274_327422


namespace dvd_cd_ratio_l3274_327409

theorem dvd_cd_ratio (total : ℕ) (dvds : ℕ) (h1 : total = 273) (h2 : dvds = 168) :
  (dvds : ℚ) / (total - dvds : ℚ) = 8 / 5 := by
  sorry

end dvd_cd_ratio_l3274_327409


namespace yvonne_swims_10_laps_l3274_327430

/-- The number of laps Yvonne can swim -/
def yvonne_laps : ℕ := sorry

/-- The number of laps Yvonne's younger sister can swim -/
def sister_laps : ℕ := sorry

/-- The number of laps Joel can swim -/
def joel_laps : ℕ := 15

theorem yvonne_swims_10_laps :
  (sister_laps = yvonne_laps / 2) →
  (joel_laps = 3 * sister_laps) →
  (yvonne_laps = 10) :=
by sorry

end yvonne_swims_10_laps_l3274_327430


namespace ceiling_negative_sqrt_64_over_9_l3274_327426

theorem ceiling_negative_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end ceiling_negative_sqrt_64_over_9_l3274_327426


namespace set_equality_l3274_327401

open Set

-- Define the sets
def R : Set ℝ := univ
def A : Set ℝ := {x | x^2 ≥ 4}
def B : Set ℝ := {y | ∃ x, y = |Real.tan x|}

-- State the theorem
theorem set_equality : (R \ A) ∩ B = {x | 0 ≤ x ∧ x < 2} := by sorry

end set_equality_l3274_327401


namespace set_operation_result_l3274_327491

def set_operation (M N : Set Int) : Set Int :=
  {x | ∃ y z, y ∈ N ∧ z ∈ M ∧ x = y - z}

theorem set_operation_result :
  let M : Set Int := {0, 1, 2}
  let N : Set Int := {-2, -3}
  set_operation M N = {-2, -3, -4, -5} := by
  sorry

end set_operation_result_l3274_327491


namespace range_of_H_l3274_327439

-- Define the function H
def H (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H : 
  Set.range H = {-1, 5} := by sorry

end range_of_H_l3274_327439


namespace subtraction_decimal_l3274_327452

theorem subtraction_decimal : (3.56 : ℝ) - (1.89 : ℝ) = 1.67 := by
  sorry

end subtraction_decimal_l3274_327452


namespace two_distinct_roots_for_all_m_m_value_when_root_sum_condition_l3274_327459

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4*eq.a*eq.c

/-- Checks if a quadratic equation has two distinct real roots -/
def has_two_distinct_real_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq > 0

/-- Our specific quadratic equation x^2 - 2x - 3m^2 = 0 -/
def our_equation (m : ℝ) : QuadraticEquation :=
  { a := 1, b := -2, c := -3*m^2 }

theorem two_distinct_roots_for_all_m (m : ℝ) :
  has_two_distinct_real_roots (our_equation m) := by
  sorry

theorem m_value_when_root_sum_condition (m : ℝ) (α β : ℝ)
  (h1 : α + β = 2)
  (h2 : α + 2*β = 5)
  (h3 : α * β = -(-3*m^2)) :
  m = 1 ∨ m = -1 := by
  sorry

end two_distinct_roots_for_all_m_m_value_when_root_sum_condition_l3274_327459


namespace sum_of_data_l3274_327412

theorem sum_of_data (a b c : ℝ) : 
  a + b = c → 
  b = 3 * a → 
  a = 12 → 
  a + b + c = 96 := by
sorry

end sum_of_data_l3274_327412


namespace sector_max_area_and_angle_l3274_327431

/-- Given a sector of a circle with perimeter 30 cm, prove that the maximum area is 225/4 cm² 
    and the corresponding central angle is 2 radians. -/
theorem sector_max_area_and_angle (r : ℝ) (l : ℝ) (α : ℝ) (area : ℝ) :
  l + 2 * r = 30 →                            -- Perimeter condition
  l = r * α →                                 -- Arc length formula
  area = (1 / 2) * r * l →                    -- Area formula for sector
  (∀ r' l' α' area', l' + 2 * r' = 30 → l' = r' * α' → area' = (1 / 2) * r' * l' → area' ≤ area) →
  area = 225 / 4 ∧ α = 2 := by
  sorry

end sector_max_area_and_angle_l3274_327431


namespace first_nonzero_digit_of_fraction_l3274_327416

theorem first_nonzero_digit_of_fraction (n : ℕ) (h : n = 1029) : 
  ∃ (k : ℕ) (d : ℕ), 
    0 < d ∧ d < 10 ∧
    (↑k : ℚ) < (1 : ℚ) / n ∧
    (1 : ℚ) / n < ((↑k + 1) : ℚ) / 10 ∧
    d = 9 :=
sorry

end first_nonzero_digit_of_fraction_l3274_327416


namespace common_roots_solution_l3274_327434

/-- Two cubic polynomials with two distinct common roots -/
def has_two_common_roots (c d : ℝ) : Prop :=
  ∃ (p q : ℝ), p ≠ q ∧
    p^3 + c*p^2 + 7*p + 4 = 0 ∧
    p^3 + d*p^2 + 10*p + 6 = 0 ∧
    q^3 + c*q^2 + 7*q + 4 = 0 ∧
    q^3 + d*q^2 + 10*q + 6 = 0

/-- The theorem stating the unique solution for c and d -/
theorem common_roots_solution :
  ∀ c d : ℝ, has_two_common_roots c d → c = -5 ∧ d = -6 :=
by sorry

end common_roots_solution_l3274_327434


namespace arithmetic_sequence_sum_first_five_l3274_327407

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

def sum_arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_first_five
  (a d : ℤ)
  (h1 : arithmetic_sequence a d 6 = 10)
  (h2 : arithmetic_sequence a d 7 = 15)
  (h3 : arithmetic_sequence a d 8 = 20) :
  sum_arithmetic_sequence a d 5 = -25 :=
by
  sorry

end arithmetic_sequence_sum_first_five_l3274_327407


namespace line_not_in_second_quadrant_l3274_327490

theorem line_not_in_second_quadrant (α : Real) (h : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi) :
  ∃ (x y : Real), x > 0 ∧ y < 0 ∧ x / Real.cos α + y / Real.sin α = 1 := by
  sorry

end line_not_in_second_quadrant_l3274_327490


namespace silly_bills_game_l3274_327476

theorem silly_bills_game (x : ℕ) : 
  x + (x + 11) + (x - 18) > 0 →  -- Ensure positive number of bills
  x + 2 * (x + 11) + 3 * (x - 18) = 100 →
  x = 22 := by
sorry

end silly_bills_game_l3274_327476


namespace fencing_cost_per_meter_l3274_327403

/-- Proves that the cost of fencing per meter for a rectangular plot with given dimensions and total fencing cost is 26.50 Rs. -/
theorem fencing_cost_per_meter
  (length breadth : ℝ)
  (length_relation : length = breadth + 10)
  (length_value : length = 55)
  (total_cost : ℝ)
  (total_cost_value : total_cost = 5300)
  : total_cost / (2 * (length + breadth)) = 26.50 := by
  sorry

end fencing_cost_per_meter_l3274_327403


namespace scientific_notation_equality_l3274_327467

theorem scientific_notation_equality : 122254 = 1.22254 * (10 ^ 5) := by
  sorry

end scientific_notation_equality_l3274_327467


namespace two_solutions_exist_l3274_327453

-- Define the function g based on the graph
noncomputable def g : ℝ → ℝ := fun x =>
  if x < -1 then -2 * x
  else if x < 3 then 2 * x + 1
  else -2 * x + 16

-- Define the property we want to prove
def satisfies_equation (x : ℝ) : Prop := g (g x) = 4

-- Theorem statement
theorem two_solutions_exist :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x, x ∈ s ↔ satisfies_equation x :=
sorry

end two_solutions_exist_l3274_327453


namespace range_of_expression_l3274_327436

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- State the theorem
theorem range_of_expression (t : AcuteTriangle) 
  (h : t.b * Real.cos t.A - t.a * Real.cos t.B = t.a) :
  2 < Real.sqrt 3 * Real.sin t.B + 2 * Real.sin t.A ^ 2 ∧ 
  Real.sqrt 3 * Real.sin t.B + 2 * Real.sin t.A ^ 2 < Real.sqrt 3 + 1 := by
  sorry

end range_of_expression_l3274_327436


namespace total_goats_is_320_l3274_327492

/-- The number of goats Washington has -/
def washington_goats : ℕ := 140

/-- The number of additional goats Paddington has compared to Washington -/
def additional_goats : ℕ := 40

/-- The total number of goats Paddington and Washington have together -/
def total_goats : ℕ := washington_goats + (washington_goats + additional_goats)

/-- Theorem stating the total number of goats is 320 -/
theorem total_goats_is_320 : total_goats = 320 := by
  sorry

end total_goats_is_320_l3274_327492


namespace sum_gcd_lcm_l3274_327445

def numbers : List Nat := [18, 24, 36]

def C : Nat := numbers.foldl Nat.gcd 0

def D : Nat := numbers.foldl Nat.lcm 1

theorem sum_gcd_lcm : C + D = 78 := by sorry

end sum_gcd_lcm_l3274_327445


namespace quadrilateral_diagonals_theorem_l3274_327498

structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : Bool

def diagonals_bisect (q : Quadrilateral) : Prop :=
  let d1 := q.vertices 2 - q.vertices 0
  let d2 := q.vertices 3 - q.vertices 1
  (d1.1 / 2 = d2.1 / 2) ∧ (d1.2 / 2 = d2.2 / 2)

def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.vertices 1 - q.vertices 0 = q.vertices 3 - q.vertices 2) ∧
  (q.vertices 2 - q.vertices 1 = q.vertices 0 - q.vertices 3)

def diagonals_equal (q : Quadrilateral) : Prop :=
  let d1 := q.vertices 2 - q.vertices 0
  let d2 := q.vertices 3 - q.vertices 1
  d1.1 * d1.1 + d1.2 * d1.2 = d2.1 * d2.1 + d2.2 * d2.2

def diagonals_perpendicular (q : Quadrilateral) : Prop :=
  let d1 := q.vertices 2 - q.vertices 0
  let d2 := q.vertices 3 - q.vertices 1
  d1.1 * d2.1 + d1.2 * d2.2 = 0

theorem quadrilateral_diagonals_theorem :
  (∀ q : Quadrilateral, diagonals_bisect q → is_parallelogram q) ∧
  (∃ q : Quadrilateral, diagonals_equal q ∧ ¬is_parallelogram q) ∧
  (∃ q : Quadrilateral, diagonals_perpendicular q ∧ ¬is_parallelogram q) ∧
  (∃ q : Quadrilateral, diagonals_equal q ∧ diagonals_perpendicular q ∧ ¬is_parallelogram q) :=
by sorry

end quadrilateral_diagonals_theorem_l3274_327498


namespace power_quotient_rule_l3274_327402

theorem power_quotient_rule (a : ℝ) : a^5 / a^3 = a^2 := by sorry

end power_quotient_rule_l3274_327402


namespace wire_cutting_l3274_327438

theorem wire_cutting (x : ℝ) :
  let total_length := Real.sqrt 600 + 12 * x
  let A := (Real.sqrt 600 + 15 * x - 9 * x^2) / 2
  let B := (Real.sqrt 600 + 9 * x - 9 * x^2) / 2
  let C := 9 * x^2
  (A = B + 3 * x) ∧
  (C = (A - B)^2) ∧
  (A + B + C = total_length) :=
by sorry

end wire_cutting_l3274_327438


namespace babysitting_earnings_l3274_327481

theorem babysitting_earnings (total : ℚ) 
  (h1 : total / 4 + total / 2 + 50 = total) : total = 200 := by
  sorry

end babysitting_earnings_l3274_327481


namespace hyperbola_sum_l3274_327499

theorem hyperbola_sum (F₁ F₂ : ℝ × ℝ) (h k a b : ℝ) :
  F₁ = (-2, 0) →
  F₂ = (2, 0) →
  a > 0 →
  b > 0 →
  (∀ P : ℝ × ℝ, |dist P F₁ - dist P F₂| = 2 ↔ 
    (P.1 - h)^2 / a^2 - (P.2 - k)^2 / b^2 = 1) →
  h + k + a + b = 1 + Real.sqrt 3 :=
by sorry

end hyperbola_sum_l3274_327499


namespace johns_remaining_money_l3274_327493

/-- Calculates the amount of money John has left after walking his dog and spending money on books and his sister. -/
theorem johns_remaining_money (total_days : Nat) (sundays : Nat) (daily_pay : Nat) (book_cost : Nat) (sister_gift : Nat) :
  total_days = 30 →
  sundays = 4 →
  daily_pay = 10 →
  book_cost = 50 →
  sister_gift = 50 →
  (total_days - sundays) * daily_pay - (book_cost + sister_gift) = 160 := by
  sorry

end johns_remaining_money_l3274_327493


namespace inscribed_circle_exists_l3274_327419

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle with center and radius -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Given configuration satisfies the problem conditions -/
def ValidConfiguration (rect : Rectangle) (circleA : Circle) (circleB : Circle) (circleC : Circle) (circleD : Circle) (e : ℝ) : Prop :=
  let a := circleA.radius
  let b := circleB.radius
  let c := circleC.radius
  let d := circleD.radius
  (a + c = b + d) ∧ (a + c < e) ∧
  (rect.A = circleA.center) ∧ (rect.B = circleB.center) ∧ (rect.C = circleC.center) ∧ (rect.D = circleD.center)

/-- Theorem: A circle can be inscribed in the quadrilateral formed by outer common tangents -/
theorem inscribed_circle_exists (rect : Rectangle) (circleA : Circle) (circleB : Circle) (circleC : Circle) (circleD : Circle) (e : ℝ) 
  (h : ValidConfiguration rect circleA circleB circleC circleD e) : 
  ∃ (inscribedCircle : Circle), true :=
sorry

end inscribed_circle_exists_l3274_327419


namespace problem_statement_l3274_327463

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a / x + x * Real.log x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

-- Define the intervals
def I : Set ℝ := Set.Icc 0 2
def J : Set ℝ := Set.Icc (1/2) 2

-- State the theorem
theorem problem_statement :
  (∃ M : ℤ, (M = 4 ∧ ∀ N : ℤ, N > M → ¬∃ x₁ x₂ : ℝ, x₁ ∈ I ∧ x₂ ∈ I ∧ g x₁ - g x₂ ≥ N)) ∧
  (∃ a : ℝ, (a = 1 ∧ ∀ s t : ℝ, s ∈ J → t ∈ J → f a s ≥ g t) ∧
            ∀ b : ℝ, b < a → ∃ s t : ℝ, s ∈ J ∧ t ∈ J ∧ f b s < g t) :=
by sorry

end

end problem_statement_l3274_327463


namespace enemies_left_undefeated_l3274_327472

theorem enemies_left_undefeated 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (points_earned : ℕ) : ℕ :=
by
  have h1 : points_per_enemy = 5 := by sorry
  have h2 : total_enemies = 8 := by sorry
  have h3 : points_earned = 10 := by sorry
  
  -- Define the number of enemies defeated
  let enemies_defeated := points_earned / points_per_enemy
  
  -- Calculate enemies left undefeated
  let enemies_left := total_enemies - enemies_defeated
  
  exact enemies_left

end enemies_left_undefeated_l3274_327472


namespace intersection_S_T_l3274_327495

def S : Set ℤ := {-4, -3, 6, 7}
def T : Set ℤ := {x | x^2 > 4*x}

theorem intersection_S_T : S ∩ T = {-4, -3, 6, 7} := by sorry

end intersection_S_T_l3274_327495


namespace min_value_expression_min_value_achievable_l3274_327418

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 3*x + 1) * (y^2 + 3*y + 1) * (z^2 + 3*z + 1) / (x*y*z) ≥ 125 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + 3*x + 1) * (y^2 + 3*y + 1) * (z^2 + 3*z + 1) / (x*y*z) = 125 :=
by sorry

end min_value_expression_min_value_achievable_l3274_327418


namespace find_F_when_C_is_35_l3274_327470

-- Define the relationship between C and F
def C_F_relation (C F : ℝ) : Prop := C = (4/7) * (F - 40)

-- State the theorem
theorem find_F_when_C_is_35 :
  ∃ F : ℝ, C_F_relation 35 F ∧ F = 101.25 := by
  sorry

end find_F_when_C_is_35_l3274_327470


namespace tetrahedron_self_dual_cube_octahedron_dual_dodecahedron_icosahedron_dual_l3274_327425

/-- A polyhedron with faces and vertices -/
structure Polyhedron where
  faces : ℕ
  vertices : ℕ
  face_sides : ℕ
  vertex_valence : ℕ

/-- Duality relation between polyhedra -/
def is_dual (p q : Polyhedron) : Prop :=
  p.faces = q.vertices ∧ p.vertices = q.faces ∧
  p.face_sides = q.vertex_valence ∧ p.vertex_valence = q.face_sides

/-- Self-duality of a polyhedron -/
def is_self_dual (p : Polyhedron) : Prop :=
  is_dual p p

/-- Theorem: Tetrahedron is self-dual -/
theorem tetrahedron_self_dual :
  is_self_dual ⟨4, 4, 3, 3⟩ := by sorry

/-- Theorem: Cube and octahedron are dual -/
theorem cube_octahedron_dual :
  is_dual ⟨6, 8, 4, 3⟩ ⟨8, 6, 3, 4⟩ := by sorry

/-- Theorem: Dodecahedron and icosahedron are dual -/
theorem dodecahedron_icosahedron_dual :
  is_dual ⟨12, 20, 5, 3⟩ ⟨20, 12, 3, 5⟩ := by sorry

end tetrahedron_self_dual_cube_octahedron_dual_dodecahedron_icosahedron_dual_l3274_327425
