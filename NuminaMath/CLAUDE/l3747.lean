import Mathlib

namespace equation_solution_l3747_374702

theorem equation_solution : ∃ x : ℤ, 45 - (28 - (x - (15 - 18))) = 57 ∧ x = 37 := by
  sorry

end equation_solution_l3747_374702


namespace pentagon_angle_sequences_l3747_374786

def is_valid_sequence (x d : ℕ) : Prop :=
  x > 0 ∧ d > 0 ∧
  x + (x+d) + (x+2*d) + (x+3*d) + (x+4*d) = 540 ∧
  x + 4*d < 120

theorem pentagon_angle_sequences :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ s : Finset (ℕ × ℕ), s.card = n ∧ 
    (∀ p ∈ s, is_valid_sequence p.1 p.2) ∧
    (∀ x d : ℕ, is_valid_sequence x d → (x, d) ∈ s)) :=
sorry

end pentagon_angle_sequences_l3747_374786


namespace sophie_donuts_left_l3747_374742

/-- Calculates the number of donuts left for Sophie after giving some away --/
def donuts_left (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_given_to_mom : ℕ) (donuts_given_to_sister : ℕ) : ℕ :=
  total_boxes * donuts_per_box - boxes_given_to_mom * donuts_per_box - donuts_given_to_sister

/-- Proves that Sophie has 30 donuts left --/
theorem sophie_donuts_left : donuts_left 4 12 1 6 = 30 := by
  sorry

end sophie_donuts_left_l3747_374742


namespace max_chickens_and_chicks_optimal_chicken_count_l3747_374770

/-- Represents the chicken coop problem -/
structure ChickenCoop where
  area : ℝ
  chicken_space : ℝ
  chick_space : ℝ
  chicken_feed : ℝ
  chick_feed : ℝ
  max_feed : ℝ

/-- Defines the conditions of the problem -/
def problem_conditions : ChickenCoop :=
  { area := 240
  , chicken_space := 4
  , chick_space := 2
  , chicken_feed := 160
  , chick_feed := 40
  , max_feed := 8000
  }

/-- Checks if a given number of chickens and chicks satisfies the constraints -/
def satisfies_constraints (coop : ChickenCoop) (chickens : ℕ) (chicks : ℕ) : Prop :=
  (chickens : ℝ) * coop.chicken_space + (chicks : ℝ) * coop.chick_space ≤ coop.area ∧
  (chickens : ℝ) * coop.chicken_feed + (chicks : ℝ) * coop.chick_feed ≤ coop.max_feed

/-- Theorem stating the maximum number of chickens and chicks -/
theorem max_chickens_and_chicks (coop : ChickenCoop := problem_conditions) :
  satisfies_constraints coop 40 40 ∧
  (∀ c : ℕ, c > 40 → ¬satisfies_constraints coop c 40) ∧
  satisfies_constraints coop 0 120 ∧
  (∀ k : ℕ, k > 120 → ¬satisfies_constraints coop 0 k) := by
  sorry

/-- Theorem stating that 40 chickens and 40 chicks is optimal when maximizing chickens -/
theorem optimal_chicken_count (coop : ChickenCoop := problem_conditions) :
  ∀ c k : ℕ, satisfies_constraints coop c k →
    c ≤ 40 ∧ (c = 40 → k ≤ 40) := by
  sorry

end max_chickens_and_chicks_optimal_chicken_count_l3747_374770


namespace cement_mixture_weight_l3747_374724

theorem cement_mixture_weight :
  ∀ (W : ℝ),
    (1/3 : ℝ) * W + (1/4 : ℝ) * W + 10 = W →
    W = 24 := by
  sorry

end cement_mixture_weight_l3747_374724


namespace min_box_value_l3747_374768

theorem min_box_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 32 * x^2 + box * x + 32) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  (∃ a' b' box', (∀ x, (a' * x + b') * (b' * x + a') = 32 * x^2 + box' * x + 32) ∧
                 a' ≠ b' ∧ a' ≠ box' ∧ b' ≠ box' ∧
                 box' ≥ 80) →
  box ≥ 80 :=
by sorry

end min_box_value_l3747_374768


namespace two_roots_implies_c_values_l3747_374711

-- Define the function f(x) = x³ - 3x + c
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

-- Define the property of having exactly two roots
def has_exactly_two_roots (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂

-- Theorem statement
theorem two_roots_implies_c_values (c : ℝ) :
  has_exactly_two_roots (f c) → c = -2 ∨ c = 2 :=
sorry

end two_roots_implies_c_values_l3747_374711


namespace supplement_of_half_angle_l3747_374798

-- Define the angle α
def α : ℝ := 90 - 50

-- Theorem statement
theorem supplement_of_half_angle (h : α = 90 - 50) : 
  180 - (α / 2) = 160 := by sorry

end supplement_of_half_angle_l3747_374798


namespace largest_k_value_l3747_374778

theorem largest_k_value (x y k : ℝ) : 
  (2 * x + y = k) →
  (3 * x + y = 3) →
  (x - 2 * y ≥ 1) →
  (∀ m : ℤ, m > k → ¬(∃ x' y' : ℝ, 2 * x' + y' = m ∧ 3 * x' + y' = 3 ∧ x' - 2 * y' ≥ 1)) →
  k ≤ 2 ∧ (∃ x' y' : ℝ, 2 * x' + y' = 2 ∧ 3 * x' + y' = 3 ∧ x' - 2 * y' ≥ 1) :=
by sorry

end largest_k_value_l3747_374778


namespace sum_of_distances_l3747_374704

/-- Parabola with equation y^2 = 4x -/
structure Parabola where
  eq : ∀ x y : ℝ, y^2 = 4*x

/-- Line with equation 2x + y - 4 = 0 -/
structure Line where
  eq : ∀ x y : ℝ, 2*x + y - 4 = 0

/-- Point A with coordinates (1, 2) -/
def A : ℝ × ℝ := (1, 2)

/-- Point B, the other intersection of the parabola and line -/
def B : ℝ × ℝ := sorry

/-- F is the focus of the parabola -/
def F : ℝ × ℝ := sorry

/-- |FA| is the distance between F and A -/
def FA : ℝ := sorry

/-- |FB| is the distance between F and B -/
def FB : ℝ := sorry

/-- Theorem stating that |FA| + |FB| = 7 -/
theorem sum_of_distances (p : Parabola) (l : Line) : FA + FB = 7 := by
  sorry

end sum_of_distances_l3747_374704


namespace exists_complementary_not_acute_not_obtuse_l3747_374710

-- Define complementary angles
def complementary (a b : ℝ) : Prop := a + b = 180

-- Define acute angle
def acute (a : ℝ) : Prop := 0 < a ∧ a < 90

-- Define obtuse angle
def obtuse (a : ℝ) : Prop := 90 < a ∧ a < 180

-- Theorem statement
theorem exists_complementary_not_acute_not_obtuse :
  ∃ (a b : ℝ), complementary a b ∧ ¬(acute a ∨ obtuse a) ∧ ¬(acute b ∨ obtuse b) :=
sorry

end exists_complementary_not_acute_not_obtuse_l3747_374710


namespace distance_to_nearest_city_l3747_374797

theorem distance_to_nearest_city (d : ℝ) : 
  (¬(d ≥ 8)) ∧ (¬(d ≤ 7)) ∧ (¬(d ≤ 6)) ∧ (d ≠ 10) → 7 < d ∧ d < 8 := by
  sorry

end distance_to_nearest_city_l3747_374797


namespace smallest_power_l3747_374736

theorem smallest_power (a b c d : ℕ) : 
  2^55 < 3^44 ∧ 2^55 < 5^33 ∧ 2^55 < 6^22 :=
by sorry

end smallest_power_l3747_374736


namespace simplest_quadratic_radical_l3747_374720

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ 
  (∀ (m : ℕ), m ^ 2 ∣ n → m = 1) ∧
  (∀ (a b : ℕ), n = a / b → b = 1)

theorem simplest_quadratic_radical :
  ¬ is_simplest_quadratic_radical (Real.sqrt (1 / 2)) ∧
  is_simplest_quadratic_radical (Real.sqrt 3) ∧
  ¬ is_simplest_quadratic_radical (Real.sqrt 8) ∧
  ¬ is_simplest_quadratic_radical (Real.sqrt 0.1) :=
by sorry

end simplest_quadratic_radical_l3747_374720


namespace sum_of_penultimate_terms_l3747_374744

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_penultimate_terms 
  (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_first : a 0 = 3) 
  (h_last : ∃ n : ℕ, a n = 33 ∧ a (n - 1) + a (n - 2) = x + y) : 
  x + y = 51 := by
sorry

end sum_of_penultimate_terms_l3747_374744


namespace min_bottles_to_fill_l3747_374749

/-- The capacity of a small bottle in milliliters -/
def small_bottle_capacity : ℝ := 35

/-- The capacity of a large bottle in milliliters -/
def large_bottle_capacity : ℝ := 500

/-- The minimum number of small bottles needed to completely fill a large bottle -/
def min_bottles : ℕ := 15

theorem min_bottles_to_fill :
  ∃ (n : ℕ), n * small_bottle_capacity ≥ large_bottle_capacity ∧
  ∀ (m : ℕ), m * small_bottle_capacity ≥ large_bottle_capacity → n ≤ m ∧
  n = min_bottles :=
by sorry

end min_bottles_to_fill_l3747_374749


namespace ellipse_properties_l3747_374796

/-- Represents an ellipse defined by the equation (x^2 / 36) + (y^2 / 9) = 4 -/
def Ellipse := {(x, y) : ℝ × ℝ | (x^2 / 36) + (y^2 / 9) = 4}

/-- The distance between the foci of the ellipse -/
def focalDistance (e : Set (ℝ × ℝ)) : ℝ := 
  5.196

/-- The eccentricity of the ellipse -/
def eccentricity (e : Set (ℝ × ℝ)) : ℝ := 
  0.866

theorem ellipse_properties : 
  focalDistance Ellipse = 5.196 ∧ eccentricity Ellipse = 0.866 := by
  sorry

end ellipse_properties_l3747_374796


namespace rectangle_width_l3747_374717

/-- Given a rectangle with area 1638 square inches, where ten such rectangles
    would have a total length of 390 inches, prove that its width is 42 inches. -/
theorem rectangle_width (area : ℝ) (total_length : ℝ) (h1 : area = 1638) 
    (h2 : total_length = 390) : ∃ (width : ℝ), width = 42 ∧ 
    ∃ (length : ℝ), area = length * width ∧ total_length = 10 * length :=
sorry

end rectangle_width_l3747_374717


namespace min_c_value_l3747_374745

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : ∃! (x y : ℝ), 2*x + y = 2029 ∧ y = |x - a| + |x - b| + |x - c|) : 
  c ≥ 1015 :=
by sorry

end min_c_value_l3747_374745


namespace intersection_A_B_l3747_374781

-- Define set A
def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}

-- Define set B
def B : Set ℝ := {y | 0 ≤ y ∧ y < 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 4} := by sorry

end intersection_A_B_l3747_374781


namespace union_cardinality_of_subset_count_l3747_374780

/-- Given two finite sets A and B, if the number of sets which are subsets of A or subsets of B is 144, then the cardinality of their union is 8. -/
theorem union_cardinality_of_subset_count (A B : Finset ℕ) : 
  (Finset.powerset A).card + (Finset.powerset B).card - (Finset.powerset (A ∩ B)).card = 144 →
  (A ∪ B).card = 8 := by
  sorry

end union_cardinality_of_subset_count_l3747_374780


namespace unique_a_value_l3747_374759

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_value (a : ℝ) (h : 1 ∈ A a) : a = 0 := by
  sorry

end unique_a_value_l3747_374759


namespace consecutive_integers_product_sum_l3747_374734

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end consecutive_integers_product_sum_l3747_374734


namespace combined_variance_is_100_l3747_374735

/-- Calculates the combined variance of two classes given their individual statistics -/
def combinedVariance (nA nB : ℕ) (meanA meanB : ℝ) (varA varB : ℝ) : ℝ :=
  let n := nA + nB
  let pA := nA / n
  let pB := nB / n
  let combinedMean := pA * meanA + pB * meanB
  pA * (varA + (meanA - combinedMean)^2) + pB * (varB + (meanB - combinedMean)^2)

/-- The variance of the combined scores of Class A and Class B is 100 -/
theorem combined_variance_is_100 :
  combinedVariance 50 40 76 85 96 60 = 100 := by
  sorry

end combined_variance_is_100_l3747_374735


namespace arkos_population_2070_l3747_374761

def population_growth (initial_population : ℕ) (start_year end_year doubling_period : ℕ) : ℕ :=
  initial_population * (2 ^ ((end_year - start_year) / doubling_period))

theorem arkos_population_2070 :
  population_growth 150 1960 2070 20 = 4800 :=
by
  sorry

end arkos_population_2070_l3747_374761


namespace continuous_function_property_l3747_374731

theorem continuous_function_property (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_prop : ∀ x y : ℝ, f (x + y) * f (x - y) = f x ^ 2) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x ≠ 0) := by
  sorry

end continuous_function_property_l3747_374731


namespace berry_exchange_theorem_l3747_374785

/-- The number of blueberries in each blue box -/
def B : ℕ := 35

/-- The number of strawberries in each red box -/
def S : ℕ := 100 + B

/-- The change in total berries when exchanging one blue box for one red box -/
def ΔT : ℤ := S - B

theorem berry_exchange_theorem : ΔT = 65 := by
  sorry

end berry_exchange_theorem_l3747_374785


namespace reciprocal_sum_theorem_l3747_374727

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 
  1 / x + 1 / y = 3 := by
sorry

end reciprocal_sum_theorem_l3747_374727


namespace cosine_product_bounds_l3747_374732

theorem cosine_product_bounds : 
  1/8 < Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (70 * π / 180) ∧ 
  Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (70 * π / 180) < 1/4 :=
by sorry

end cosine_product_bounds_l3747_374732


namespace min_value_theorem_equality_condition_l3747_374739

theorem min_value_theorem (x : ℝ) (h : x > 0) : 4 * x + 1 / x^4 ≥ 5 := by
  sorry

theorem equality_condition : 4 * 1 + 1 / 1^4 = 5 := by
  sorry

end min_value_theorem_equality_condition_l3747_374739


namespace sin_2alpha_proof_l3747_374737

theorem sin_2alpha_proof (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end sin_2alpha_proof_l3747_374737


namespace rectangular_prism_sum_l3747_374790

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- Properties of a rectangular prism -/
axiom rectangular_prism_properties (rp : RectangularPrism) : 
  rp.faces = 6 ∧ rp.edges = 12 ∧ rp.vertices = 8

/-- Theorem: The sum of faces, edges, and vertices of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end rectangular_prism_sum_l3747_374790


namespace least_multiple_of_29_above_500_l3747_374700

theorem least_multiple_of_29_above_500 : 
  ∀ n : ℕ, n > 0 ∧ 29 ∣ n ∧ n > 500 → n ≥ 522 := by
  sorry

end least_multiple_of_29_above_500_l3747_374700


namespace simplify_abs_sum_l3747_374794

def second_quadrant (a b : ℝ) : Prop := a < 0 ∧ b > 0

theorem simplify_abs_sum (a b : ℝ) (h : second_quadrant a b) : 
  |a - b| + |b - a| = -2*a + 2*b := by
sorry

end simplify_abs_sum_l3747_374794


namespace activity_participation_l3747_374707

def total_sample : ℕ := 100
def male_participants : ℕ := 60
def willing_to_participate : ℕ := 70
def males_willing : ℕ := 48
def females_not_willing : ℕ := 18

def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

def critical_value : ℚ := 6635 / 1000

theorem activity_participation :
  let females_willing := willing_to_participate - males_willing
  let males_not_willing := male_participants - males_willing
  let female_participants := total_sample - male_participants
  let chi_sq := chi_square males_willing females_willing males_not_willing females_not_willing
  let male_proportion := (males_willing : ℚ) / male_participants
  let female_proportion := (females_willing : ℚ) / female_participants
  (chi_sq > critical_value) ∧
  (male_proportion > female_proportion) ∧
  (12 / 7 : ℚ) = (4 * 0 + 3 * 1 + 2 * 2 + 1 * 3 : ℚ) / (Nat.choose 7 3) := by sorry

end activity_participation_l3747_374707


namespace johns_tax_rate_johns_tax_rate_approx_30_percent_l3747_374701

/-- Calculates John's tax rate given the incomes and tax rates of John and Ingrid --/
theorem johns_tax_rate (john_income ingrid_income : ℝ) 
                       (ingrid_tax_rate combined_tax_rate : ℝ) : ℝ :=
  let total_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * total_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let john_tax := total_tax - ingrid_tax
  john_tax / john_income

/-- John's tax rate is approximately 30.00% --/
theorem johns_tax_rate_approx_30_percent : 
  ∃ ε > 0, 
    |johns_tax_rate 58000 72000 0.40 0.3554 - 0.30| < ε ∧ 
    ε < 0.0001 :=
sorry

end johns_tax_rate_johns_tax_rate_approx_30_percent_l3747_374701


namespace f_derivative_condition_implies_a_range_g_minimum_value_l3747_374733

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + ((a-3)/2) * x^2 + (a^2-3*a) * x - 2*a

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-3)*x + a^2 - 3*a

-- Define the function g
def g (a x₁ x₂ : ℝ) : ℝ := x₁^3 + x₂^3 + a^3

theorem f_derivative_condition_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f_derivative a x > a^2) → a ∈ Set.Ioi (-2) :=
sorry

theorem g_minimum_value (a x₁ x₂ : ℝ) :
  a ∈ Set.Ioo (-1) 3 →
  x₁ ≠ x₂ →
  f_derivative a x₁ = 0 →
  f_derivative a x₂ = 0 →
  g a x₁ x₂ ≥ 15 :=
sorry

end

end f_derivative_condition_implies_a_range_g_minimum_value_l3747_374733


namespace stream_speed_l3747_374721

/-- Proves that the speed of a stream is 4 km/hr given the conditions of the boat's travel. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 13 →
  downstream_distance = 68 →
  downstream_time = 4 →
  (boat_speed + (downstream_distance / downstream_time - boat_speed)) = 17 := by
  sorry

#check stream_speed

end stream_speed_l3747_374721


namespace polynomial_intersection_l3747_374755

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem polynomial_intersection (a b c d : ℝ) : 
  -- f and g are distinct
  f a b ≠ g c d →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- Both f and g have the same minimum value
  (∃ (k : ℝ), ∀ (x : ℝ), f a b x ≥ k ∧ g c d x ≥ k) →
  -- The graphs of f and g intersect at (200, -200)
  f a b 200 = -200 ∧ g c d 200 = -200 →
  -- Conclusion: a + c = -800
  a + c = -800 := by
sorry

end polynomial_intersection_l3747_374755


namespace license_plate_combinations_l3747_374793

/-- The number of letters in the English alphabet -/
def alphabet_count : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of consonants (including Y) -/
def consonant_count : ℕ := alphabet_count - vowel_count

/-- The number of digits -/
def digit_count : ℕ := 10

/-- The total number of possible license plates -/
def license_plate_count : ℕ := consonant_count * vowel_count * consonant_count * digit_count

theorem license_plate_combinations : license_plate_count = 22050 := by
  sorry

end license_plate_combinations_l3747_374793


namespace probability_five_diamond_ace_l3747_374776

-- Define the structure of a standard deck
def StandardDeck : Type := Fin 52

-- Define card properties
def isFive (card : StandardDeck) : Prop := sorry
def isDiamond (card : StandardDeck) : Prop := sorry
def isAce (card : StandardDeck) : Prop := sorry

-- Define the probability of drawing three specific cards
def probabilityOfDraw (deck : Type) (pred1 pred2 pred3 : deck → Prop) : ℚ := sorry

-- Theorem statement
theorem probability_five_diamond_ace :
  probabilityOfDraw StandardDeck isFive isDiamond isAce = 85 / 44200 := by
  sorry

end probability_five_diamond_ace_l3747_374776


namespace parabola_ellipse_focus_coincidence_l3747_374792

/-- The value of p for which the focus of the parabola y² = 2px coincides with 
    the right focus of the ellipse x²/5 + y² = 1 -/
theorem parabola_ellipse_focus_coincidence : ∃ p : ℝ, 
  (∀ x y : ℝ, y^2 = 2*p*x → x^2/5 + y^2 = 1 → x = 2) → p = 4 := by
  sorry

end parabola_ellipse_focus_coincidence_l3747_374792


namespace row_sum_equals_square_l3747_374775

theorem row_sum_equals_square (k : ℕ) (h : k > 0) : 
  let n := 2 * k - 1
  let a := k
  let l := 3 * k - 2
  (n * (a + l)) / 2 = (2 * k - 1)^2 := by
sorry

end row_sum_equals_square_l3747_374775


namespace original_loaf_size_l3747_374747

def slices_per_sandwich : ℕ := 2
def days_with_one_sandwich : ℕ := 5
def sandwiches_on_saturday : ℕ := 2
def slices_left : ℕ := 6

theorem original_loaf_size :
  slices_per_sandwich * days_with_one_sandwich +
  slices_per_sandwich * sandwiches_on_saturday +
  slices_left = 20 := by
  sorry

end original_loaf_size_l3747_374747


namespace negative_three_to_fourth_power_l3747_374773

theorem negative_three_to_fourth_power :
  -3^4 = -(3 * 3 * 3 * 3) := by sorry

end negative_three_to_fourth_power_l3747_374773


namespace focus_of_parabola_l3747_374757

/-- The focus of the parabola x = -1/4 * y^2 -/
def parabola_focus : ℝ × ℝ := (-1, 0)

/-- The equation of the parabola -/
def is_on_parabola (x y : ℝ) : Prop := x = -1/4 * y^2

/-- Theorem stating that the focus of the parabola x = -1/4 * y^2 is at (-1, 0) -/
theorem focus_of_parabola :
  let (f, g) := parabola_focus
  ∀ (x y : ℝ), is_on_parabola x y →
    (x - f)^2 + y^2 = (x - (-f))^2 :=
by sorry

end focus_of_parabola_l3747_374757


namespace some_number_in_formula_l3747_374746

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (x : ℕ) (n : ℚ) : ℚ := 2.5 + 0.5 * (x - n)

/-- The number of axles on an 18-wheel truck -/
def axles_18_wheel_truck : ℕ := 5

/-- The toll for an 18-wheel truck -/
def toll_18_wheel_truck : ℚ := 4

theorem some_number_in_formula : 
  ∃ n : ℚ, toll_formula axles_18_wheel_truck n = toll_18_wheel_truck ∧ n = 2 :=
by sorry

end some_number_in_formula_l3747_374746


namespace train_length_l3747_374715

/-- Calculates the length of a train given its speed, the speed of a motorbike it overtakes, and the time it takes to overtake. -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) : 
  train_speed = 100 → 
  motorbike_speed = 64 → 
  overtake_time = 20 → 
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 200 := by
  sorry

#check train_length

end train_length_l3747_374715


namespace sum_of_roots_quadratic_l3747_374771

theorem sum_of_roots_quadratic (b : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ + b = 0) → (x₂^2 - 2*x₂ + b = 0) → x₁ + x₂ = 2 := by
  sorry

end sum_of_roots_quadratic_l3747_374771


namespace digit_150_is_3_l3747_374784

/-- The decimal representation of 1/13 -/
def decimal_rep_1_13 : ℕ → Fin 10
  | n => Fin.ofNat ((10 * (10^n % 13)) / 13)

/-- The length of the repeating block in the decimal representation of 1/13 -/
def rep_length : ℕ := 6

/-- The 150th digit after the decimal point in the decimal representation of 1/13 -/
def digit_150 : Fin 10 := decimal_rep_1_13 149

theorem digit_150_is_3 : digit_150 = 3 := by sorry

end digit_150_is_3_l3747_374784


namespace complex_square_sum_l3747_374774

theorem complex_square_sum (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  (a + b * i) ^ 2 = 3 + 4 * i → a ^ 2 + b ^ 2 = 5 := by
  sorry

end complex_square_sum_l3747_374774


namespace max_value_z_plus_x_l3747_374767

theorem max_value_z_plus_x :
  ∀ x y z t : ℝ,
  x^2 + y^2 = 4 →
  z^2 + t^2 = 9 →
  x*t + y*z ≥ 6 →
  z + x ≤ 5 :=
by
  sorry

end max_value_z_plus_x_l3747_374767


namespace inverse_contrapositive_l3747_374772

theorem inverse_contrapositive (x y : ℝ) : x = 0 ∧ y = 2 → x + y = 2 := by
  sorry

end inverse_contrapositive_l3747_374772


namespace descent_problem_l3747_374754

/-- The number of floors Austin and Jake descended. -/
def floors : ℕ := sorry

/-- The number of steps Jake descends per second. -/
def steps_per_second : ℕ := 3

/-- The number of steps per floor. -/
def steps_per_floor : ℕ := 30

/-- The time (in seconds) it takes Austin to reach the ground floor using the elevator. -/
def austin_time : ℕ := 60

/-- The time (in seconds) it takes Jake to reach the ground floor using the stairs. -/
def jake_time : ℕ := 90

theorem descent_problem :
  floors = (jake_time * steps_per_second) / steps_per_floor := by
  sorry

end descent_problem_l3747_374754


namespace meter_to_skips_l3747_374728

theorem meter_to_skips 
  (b c d e f g : ℝ) 
  (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) (hg : g > 0)
  (hop_skip : b * 1 = c * 1)  -- b hops = c skips
  (jump_hop : d * 1 = e * 1)  -- d jumps = e hops
  (jump_meter : f * 1 = g * 1)  -- f jumps = g meters
  : 1 = (c * e * f) / (b * d * g) := by
  sorry

end meter_to_skips_l3747_374728


namespace waiter_new_customers_l3747_374760

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 47) 
  (h2 : customers_left = 41) 
  (h3 : final_customers = 26) : 
  final_customers - (initial_customers - customers_left) = 20 :=
by sorry

end waiter_new_customers_l3747_374760


namespace four_stamps_cost_l3747_374709

/-- The cost of a single stamp in dollars -/
def stamp_cost : ℚ := 34/100

/-- The cost of two stamps in dollars -/
def two_stamps_cost : ℚ := 68/100

/-- The cost of three stamps in dollars -/
def three_stamps_cost : ℚ := 102/100

/-- Proves that the cost of four stamps is $1.36 -/
theorem four_stamps_cost :
  stamp_cost * 4 = 136/100 :=
by sorry

end four_stamps_cost_l3747_374709


namespace not_even_if_unequal_l3747_374730

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem not_even_if_unequal :
  f (-2) ≠ f 2 → ¬(IsEven f) := by
  sorry

end not_even_if_unequal_l3747_374730


namespace f_monotone_increasing_f_monotone_decreasing_f_not_always_above_a_l3747_374706

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- Theorem 1: f(x) is monotonically increasing on ℝ iff a ≤ 0
theorem f_monotone_increasing (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 0 :=
sorry

-- Theorem 2: f(x) is monotonically decreasing on (-1, 1) iff a ≥ 3
theorem f_monotone_decreasing (a : ℝ) :
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x > f a y) ↔ a ≥ 3 :=
sorry

-- Theorem 3: ∃x ∈ ℝ, f(x) < a
theorem f_not_always_above_a (a : ℝ) :
  ∃ x : ℝ, f a x < a :=
sorry

end f_monotone_increasing_f_monotone_decreasing_f_not_always_above_a_l3747_374706


namespace sum_of_integers_l3747_374766

theorem sum_of_integers (a b : ℕ+) (h1 : a.val^2 - b.val^2 = 44) (h2 : a.val * b.val = 120) : 
  a.val + b.val = 22 := by
sorry

end sum_of_integers_l3747_374766


namespace inequality_solution_set_l3747_374712

def solution_set (x : ℝ) : Prop := x ∈ Set.Ici 0 ∩ Set.Iio 2

theorem inequality_solution_set :
  ∀ x : ℝ, x ≠ 2 → (x / (x - 2) ≤ 0 ↔ solution_set x) :=
by sorry

end inequality_solution_set_l3747_374712


namespace basketball_score_calculation_l3747_374729

/-- Given a basketball player who made 7 two-point shots and 3 three-point shots,
    the total points scored is equal to 23. -/
theorem basketball_score_calculation (two_point_shots three_point_shots : ℕ) : 
  two_point_shots = 7 →
  three_point_shots = 3 →
  2 * two_point_shots + 3 * three_point_shots = 23 := by
  sorry

end basketball_score_calculation_l3747_374729


namespace project_hours_difference_l3747_374719

/-- Given a project where three people charged time, prove that one person charged 100 more hours than another. -/
theorem project_hours_difference (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 180 ∧ 
  pat_hours = 2 * kate_hours ∧ 
  pat_hours * 3 = mark_hours ∧
  total_hours = kate_hours + pat_hours + mark_hours →
  mark_hours = kate_hours + 100 := by
  sorry

end project_hours_difference_l3747_374719


namespace weight_replacement_l3747_374722

theorem weight_replacement (initial_count : ℕ) (average_increase : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  average_increase = 2.5 →
  new_weight = 80 →
  ∃ (old_weight : ℝ),
    old_weight = new_weight - (initial_count * average_increase) ∧
    old_weight = 60 :=
by sorry

end weight_replacement_l3747_374722


namespace batsman_average_after_17th_inning_l3747_374716

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  inningsPlayed : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.inningsPlayed + 1)

/-- Theorem: A batsman's average after 17th inning is 39, given the conditions -/
theorem batsman_average_after_17th_inning
  (stats : BatsmanStats)
  (h1 : stats.inningsPlayed = 16)
  (h2 : newAverage stats 87 = stats.average + 3)
  : newAverage stats 87 = 39 := by
  sorry


end batsman_average_after_17th_inning_l3747_374716


namespace y1_greater_than_y2_l3747_374708

/-- A linear function f(x) = -x + 1 -/
def f (x : ℝ) : ℝ := -x + 1

theorem y1_greater_than_y2 (y₁ y₂ : ℝ) 
  (h₁ : f (-1) = y₁) 
  (h₂ : f 2 = y₂) : 
  y₁ > y₂ := by
  sorry

end y1_greater_than_y2_l3747_374708


namespace sum_even_factors_1176_l3747_374752

def sum_even_factors (n : ℕ) : ℕ := sorry

theorem sum_even_factors_1176 : sum_even_factors 1176 = 3192 := by sorry

end sum_even_factors_1176_l3747_374752


namespace binomial_12_9_l3747_374782

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end binomial_12_9_l3747_374782


namespace complex_multiplication_l3747_374743

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_multiplication :
  i * (2 - i) = 1 + 2*i :=
by sorry

end complex_multiplication_l3747_374743


namespace dennis_teaching_years_l3747_374726

/-- Given that Virginia, Adrienne, and Dennis have taught history for a combined total of 93 years,
    Virginia has taught for 9 more years than Adrienne, and Virginia has taught for 9 fewer years than Dennis,
    prove that Dennis has taught for 40 years. -/
theorem dennis_teaching_years (v a d : ℕ) : 
  v + a + d = 93 →
  v = a + 9 →
  d = v + 9 →
  d = 40 := by
sorry

end dennis_teaching_years_l3747_374726


namespace problem_solution_l3747_374763

def f (x a : ℝ) : ℝ := |2*x - a| + |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x (-1) ≤ 2 ↔ -1/2 ≤ x ∧ x ≤ 1/2) ∧
  ((∀ x : ℝ, 1/2 ≤ x ∧ x ≤ 1 → f x a ≤ |2*x + 1|) → 0 ≤ a ∧ a ≤ 3) :=
sorry

end problem_solution_l3747_374763


namespace smallest_four_digit_congruent_to_one_mod_seventeen_l3747_374762

theorem smallest_four_digit_congruent_to_one_mod_seventeen :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 17 = 1 → n ≥ 1003 :=
by sorry

end smallest_four_digit_congruent_to_one_mod_seventeen_l3747_374762


namespace initial_milk_percentage_l3747_374753

/-- Given a mixture of milk and water, prove that the initial milk percentage is 84% -/
theorem initial_milk_percentage 
  (initial_volume : ℝ) 
  (added_water : ℝ) 
  (final_milk_percentage : ℝ) :
  initial_volume = 60 →
  added_water = 14.117647058823536 →
  final_milk_percentage = 68 →
  (initial_volume * (84 / 100)) / (initial_volume + added_water) = final_milk_percentage / 100 :=
by sorry

end initial_milk_percentage_l3747_374753


namespace geometric_sequence_problem_l3747_374769

/-- Given a geometric sequence with common ratio 2 and all positive terms,
    if the product of the 4th and 12th terms is 64, then the 7th term is 4. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio is 2
  (∀ n, a n > 0) →              -- All terms are positive
  a 4 * a 12 = 64 →             -- Product of 4th and 12th terms is 64
  a 7 = 4 := by                 -- The 7th term is 4
sorry

end geometric_sequence_problem_l3747_374769


namespace tickets_sold_and_given_away_l3747_374748

theorem tickets_sold_and_given_away (initial_tickets : ℕ) (h : initial_tickets = 5760) :
  let sold_tickets := initial_tickets / 2
  let remaining_tickets := initial_tickets - sold_tickets
  let given_away_tickets := remaining_tickets / 4
  sold_tickets + given_away_tickets = 3600 :=
by sorry

end tickets_sold_and_given_away_l3747_374748


namespace min_group_size_l3747_374714

theorem min_group_size (adult_group_size children_group_size : ℕ) 
  (h1 : adult_group_size = 17)
  (h2 : children_group_size = 15)
  (h3 : ∃ n : ℕ, n > 0 ∧ n % adult_group_size = 0 ∧ n % children_group_size = 0) :
  (Nat.lcm adult_group_size children_group_size = 255) := by
  sorry

end min_group_size_l3747_374714


namespace arithmetic_geometric_ratio_l3747_374751

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y * y = x * z

/-- The common ratio of a geometric sequence -/
def common_ratio (x y : ℚ) : ℚ :=
  y / x

theorem arithmetic_geometric_ratio 
  (a : ℕ → ℚ) (d : ℚ) 
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence (a 1) (a 3) (a 4)) :
  (common_ratio (a 1) (a 3) = 1/2) ∨ (common_ratio (a 1) (a 3) = 1) :=
sorry

end arithmetic_geometric_ratio_l3747_374751


namespace lawn_mowing_l3747_374789

theorem lawn_mowing (total_time : ℝ) (worked_time : ℝ) :
  total_time = 6 →
  worked_time = 3 →
  1 - (worked_time / total_time) = (1 : ℝ) / 2 := by
  sorry

end lawn_mowing_l3747_374789


namespace manufacturing_department_percentage_l3747_374765

theorem manufacturing_department_percentage (total_degrees : ℝ) (manufacturing_degrees : ℝ) :
  total_degrees = 360 →
  manufacturing_degrees = 216 →
  (manufacturing_degrees / total_degrees) * 100 = 60 := by
  sorry

end manufacturing_department_percentage_l3747_374765


namespace person_a_work_time_l3747_374740

theorem person_a_work_time (b : ℝ) (combined_rate : ℝ) (combined_time : ℝ) 
  (hb : b = 45)
  (hcombined : combined_rate * combined_time = 1 / 9)
  (htime : combined_time = 2) :
  ∃ a : ℝ, a = 30 ∧ combined_rate = 1 / a + 1 / b := by
  sorry

end person_a_work_time_l3747_374740


namespace orthographic_projection_area_l3747_374791

/-- The area of the orthographic projection of an equilateral triangle -/
theorem orthographic_projection_area (side_length : ℝ) (h : side_length = 2) :
  let original_area := (Real.sqrt 3 / 4) * side_length ^ 2
  let projection_area := (Real.sqrt 2 / 4) * original_area
  projection_area = Real.sqrt 6 / 4 := by
  sorry

end orthographic_projection_area_l3747_374791


namespace max_log_sum_l3747_374713

theorem max_log_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hsum : a + 2*b = 6) :
  ∃ (max : ℝ), max = 3 * Real.log 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 6 → Real.log x + 2 * Real.log y ≤ max :=
by sorry

end max_log_sum_l3747_374713


namespace negation_of_at_most_one_obtuse_l3747_374718

/-- Represents a triangle -/
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

/-- An angle is obtuse if it's greater than 90 degrees -/
def is_obtuse (angle : ℝ) : Prop := angle > 90

/-- At most one interior angle is obtuse -/
def at_most_one_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) → ¬is_obtuse (t.angles 1) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 1) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 2) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 1))

/-- At least two interior angles are obtuse -/
def at_least_two_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 1)) ∨
  (is_obtuse (t.angles 1) ∧ is_obtuse (t.angles 2)) ∨
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 2))

/-- The main theorem: the negation of "at most one obtuse" is "at least two obtuse" -/
theorem negation_of_at_most_one_obtuse (t : Triangle) :
  ¬(at_most_one_obtuse t) ↔ at_least_two_obtuse t :=
sorry

end negation_of_at_most_one_obtuse_l3747_374718


namespace nancy_boots_count_l3747_374764

theorem nancy_boots_count :
  ∀ (B : ℕ),
  (∃ (S H : ℕ),
    S = B + 9 ∧
    H = 3 * (S + B) ∧
    2 * B + 2 * S + 2 * H = 168) →
  B = 6 := by
sorry

end nancy_boots_count_l3747_374764


namespace seventh_term_is_4374_l3747_374741

/-- A geometric sequence of positive integers with first term 6 and fifth term 486 -/
def GeometricSequence : ℕ → ℕ :=
  fun n => 6 * (486 / 6) ^ ((n - 1) / 4)

/-- The seventh term of the geometric sequence is 4374 -/
theorem seventh_term_is_4374 : GeometricSequence 7 = 4374 := by
  sorry

end seventh_term_is_4374_l3747_374741


namespace isosceles_triangle_m_value_l3747_374738

/-- An isosceles triangle with side length 8 and other sides as roots of x^2 - 10x + m = 0 -/
structure IsoscelesTriangle where
  m : ℝ
  BC : ℝ
  AB_AC_eq : x^2 - 10*x + m = 0 → x = AB ∨ x = AC
  BC_eq : BC = 8
  isosceles : AB = AC

/-- The value of m in the isosceles triangle is either 25 or 16 -/
theorem isosceles_triangle_m_value (t : IsoscelesTriangle) : t.m = 25 ∨ t.m = 16 := by
  sorry

end isosceles_triangle_m_value_l3747_374738


namespace f_is_quadratic_l3747_374783

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 3x² + 1 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l3747_374783


namespace prob_hit_twice_in_three_shots_l3747_374705

/-- The probability of hitting a target exactly twice in three independent shots -/
theorem prob_hit_twice_in_three_shots 
  (p1 : Real) (p2 : Real) (p3 : Real)
  (h1 : p1 = 0.4) (h2 : p2 = 0.5) (h3 : p3 = 0.7) :
  p1 * p2 * (1 - p3) + (1 - p1) * p2 * p3 + p1 * (1 - p2) * p3 = 0.41 := by
sorry

end prob_hit_twice_in_three_shots_l3747_374705


namespace hanks_reading_time_l3747_374725

/-- Represents Hank's weekly reading habits -/
structure ReadingHabits where
  weekdayMorningMinutes : ℕ
  weekdayEveningMinutes : ℕ
  weekdayDays : ℕ
  weekendMultiplier : ℕ

/-- Calculates the total reading time in minutes for a week -/
def totalReadingTime (habits : ReadingHabits) : ℕ :=
  let weekdayTotal := habits.weekdayDays * (habits.weekdayMorningMinutes + habits.weekdayEveningMinutes)
  let weekendDays := 7 - habits.weekdayDays
  let weekendTotal := weekendDays * habits.weekendMultiplier * (habits.weekdayMorningMinutes + habits.weekdayEveningMinutes)
  weekdayTotal + weekendTotal

/-- Theorem stating that Hank's total reading time in a week is 810 minutes -/
theorem hanks_reading_time :
  let hanksHabits : ReadingHabits := {
    weekdayMorningMinutes := 30,
    weekdayEveningMinutes := 60,
    weekdayDays := 5,
    weekendMultiplier := 2
  }
  totalReadingTime hanksHabits = 810 := by
  sorry


end hanks_reading_time_l3747_374725


namespace simplify_expression_l3747_374787

theorem simplify_expression : 0.2 * 0.4 + 0.6 * 0.8 = 0.56 := by
  sorry

end simplify_expression_l3747_374787


namespace complementary_angles_ratio_l3747_374750

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  b / a = 2 / 7 →  -- ratio of angles is 2:7
  a = 110 :=  -- complement of larger angle is 110°
by sorry

end complementary_angles_ratio_l3747_374750


namespace article_original_price_l3747_374703

/-- Given an article sold with an 18% profit resulting in a profit of 542.8,
    prove that the original price of the article was 3016. -/
theorem article_original_price (profit_percentage : ℝ) (profit : ℝ) (original_price : ℝ) :
  profit_percentage = 18 →
  profit = 542.8 →
  profit = original_price * (profit_percentage / 100) →
  original_price = 3016 := by
  sorry

end article_original_price_l3747_374703


namespace open_box_volume_l3747_374723

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_length : ℝ) 
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 7) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 5236 := by
  sorry

end open_box_volume_l3747_374723


namespace CD_distance_l3747_374795

-- Define the points on a line
variable (A B C D : ℝ)

-- Define the order of points on the line
axiom order : A ≤ B ∧ B ≤ C ∧ C ≤ D

-- Define the given distances
axiom AB_dist : B - A = 2
axiom AC_dist : C - A = 5
axiom BD_dist : D - B = 6

-- Theorem to prove
theorem CD_distance : D - C = 3 := by
  sorry

end CD_distance_l3747_374795


namespace smallest_root_quadratic_l3747_374779

theorem smallest_root_quadratic (x : ℝ) :
  (9 * x^2 - 45 * x + 50 = 0) →
  (∀ y : ℝ, 9 * y^2 - 45 * y + 50 = 0 → x ≤ y) →
  x = 5/3 := by
  sorry

end smallest_root_quadratic_l3747_374779


namespace quadratic_minimum_l3747_374799

theorem quadratic_minimum (b : ℝ) : 
  ∃ (min : ℝ), (∀ x : ℝ, (1/2) * x^2 + 5*x - 3 ≥ (1/2) * min^2 + 5*min - 3) ∧ min = -5 :=
sorry

end quadratic_minimum_l3747_374799


namespace square_division_perimeter_paradox_l3747_374788

theorem square_division_perimeter_paradox :
  ∃ (a : ℚ) (x : ℚ), 0 < x ∧ x < a ∧ 
    (2 * (a + x)).isInt ∧ 
    (2 * (2 * a - x)).isInt ∧ 
    ¬(4 * a).isInt := by
  sorry

end square_division_perimeter_paradox_l3747_374788


namespace arithmetic_simplification_l3747_374758

theorem arithmetic_simplification : 2000 - 80 + 200 - 120 = 2000 := by
  sorry

end arithmetic_simplification_l3747_374758


namespace coefficient_x6_is_180_l3747_374756

/-- The coefficient of x^6 in the binomial expansion of (x - 2/x)^10 -/
def coefficient_x6 : ℤ := 
  let n : ℕ := 10
  let k : ℕ := (n - 6) / 2
  (n.choose k) * (-2)^k

theorem coefficient_x6_is_180 : coefficient_x6 = 180 := by
  sorry

end coefficient_x6_is_180_l3747_374756


namespace full_merit_scholarship_percentage_l3747_374777

theorem full_merit_scholarship_percentage
  (total_students : ℕ)
  (half_merit_percentage : ℚ)
  (no_scholarship_count : ℕ)
  (h1 : total_students = 300)
  (h2 : half_merit_percentage = 1 / 10)
  (h3 : no_scholarship_count = 255) :
  (total_students - (half_merit_percentage * total_students).floor - no_scholarship_count) / total_students = 1 / 20 := by
sorry

end full_merit_scholarship_percentage_l3747_374777
