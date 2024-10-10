import Mathlib

namespace water_pouring_problem_l858_85894

def water_remaining (n : ℕ) : ℚ :=
  1 / (n + 1)

theorem water_pouring_problem :
  ∃ n : ℕ, water_remaining n = 1 / 10 ∧ n = 9 := by
  sorry

end water_pouring_problem_l858_85894


namespace john_gum_purchase_l858_85872

/-- The number of packs of gum John bought -/
def num_gum_packs : ℕ := 2

/-- The number of candy bars John bought -/
def num_candy_bars : ℕ := 3

/-- The cost of one candy bar in dollars -/
def candy_bar_cost : ℚ := 3/2

/-- The total amount John paid in dollars -/
def total_paid : ℚ := 6

/-- The cost of one pack of gum in dollars -/
def gum_pack_cost : ℚ := candy_bar_cost / 2

theorem john_gum_purchase :
  num_gum_packs * gum_pack_cost + num_candy_bars * candy_bar_cost = total_paid :=
by sorry

end john_gum_purchase_l858_85872


namespace arithmetic_sequence_common_difference_l858_85876

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Main theorem -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h1 : S seq 5 = seq.a 8 + 5)
  (h2 : S seq 6 = seq.a 7 + seq.a 9 - 5) :
  seq.d = -55 / 19 := by
  sorry

end arithmetic_sequence_common_difference_l858_85876


namespace surface_area_volume_incomparable_l858_85896

-- Define the edge length of the cube
def edge_length : ℝ := 6

-- Define the surface area of the cube
def surface_area (l : ℝ) : ℝ := 6 * l^2

-- Define the volume of the cube
def volume (l : ℝ) : ℝ := l^3

-- Theorem stating that surface area and volume are incomparable
theorem surface_area_volume_incomparable :
  ¬(∃ (ord : ℝ → ℝ → Prop), 
    (∀ a b, ord a b ∨ ord b a) ∧ 
    (∀ a b c, ord a b → ord b c → ord a c) ∧
    (∀ a b, ord a b → ord b a → a = b) ∧
    (ord (surface_area edge_length) (volume edge_length) ∨ 
     ord (volume edge_length) (surface_area edge_length))) :=
by sorry


end surface_area_volume_incomparable_l858_85896


namespace zero_to_zero_undefined_l858_85880

theorem zero_to_zero_undefined : ¬ ∃ (x : ℝ), 0^(0 : ℝ) = x := by
  sorry

end zero_to_zero_undefined_l858_85880


namespace circle_radius_is_four_l858_85854

theorem circle_radius_is_four (r : ℝ) (h : 2 * (2 * Real.pi * r) = Real.pi * r^2) : r = 4 := by
  sorry

end circle_radius_is_four_l858_85854


namespace sequence_sum_l858_85816

def geometric_sequence (a : ℕ → ℚ) (r : ℚ) :=
  ∀ n, a (n + 1) = a n * r

theorem sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  geometric_sequence a r →
  a 6 = 4 →
  a 7 = 1 →
  a 4 + a 5 = 80 :=
sorry

end sequence_sum_l858_85816


namespace average_weight_increase_l858_85806

theorem average_weight_increase 
  (n : ℕ) 
  (old_weight new_weight : ℝ) 
  (h1 : n = 8)
  (h2 : old_weight = 35)
  (h3 : new_weight = 55) :
  (new_weight - old_weight) / n = 2.5 := by
  sorry

end average_weight_increase_l858_85806


namespace journey_time_increase_l858_85801

theorem journey_time_increase (total_distance : ℝ) (first_half_speed : ℝ) (overall_speed : ℝ)
  (h1 : total_distance = 640)
  (h2 : first_half_speed = 80)
  (h3 : overall_speed = 40) :
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let total_time := total_distance / overall_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time = 2 := by
  sorry

end journey_time_increase_l858_85801


namespace cubic_equation_value_l858_85871

theorem cubic_equation_value (m : ℝ) (h : m^2 + m - 1 = 0) : 
  m^3 + 2*m^2 - 2005 = -2004 := by
sorry

end cubic_equation_value_l858_85871


namespace collapsible_iff_power_of_two_l858_85840

/-- A token arrangement in the plane -/
structure TokenArrangement :=
  (n : ℕ+)  -- number of tokens
  (positions : Fin n → ℝ × ℝ)  -- positions of tokens in the plane

/-- Predicate for an arrangement being collapsible -/
def Collapsible (arrangement : TokenArrangement) : Prop :=
  ∃ (final_pos : ℝ × ℝ), ∀ i : Fin arrangement.n, 
    ∃ (moves : ℕ), arrangement.positions i = final_pos

/-- Predicate for a number being a power of 2 -/
def IsPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- The main theorem -/
theorem collapsible_iff_power_of_two :
  ∀ n : ℕ+, (∀ arrangement : TokenArrangement, arrangement.n = n → Collapsible arrangement) ↔ IsPowerOfTwo n :=
sorry

end collapsible_iff_power_of_two_l858_85840


namespace chef_lunch_meals_l858_85833

theorem chef_lunch_meals (meals_sold_lunch : ℕ) (meals_prepared_dinner : ℕ) (total_dinner_meals : ℕ)
  (h1 : meals_sold_lunch = 12)
  (h2 : meals_prepared_dinner = 5)
  (h3 : total_dinner_meals = 10) :
  meals_sold_lunch + (total_dinner_meals - meals_prepared_dinner) = 17 :=
by sorry

end chef_lunch_meals_l858_85833


namespace pascal_triangle_101_row_third_number_l858_85803

/-- The number of elements in a row of Pascal's triangle -/
def row_elements (n : ℕ) : ℕ := n + 1

/-- The third number in a row of Pascal's triangle -/
def third_number (n : ℕ) : ℕ := n.choose 2

theorem pascal_triangle_101_row_third_number :
  ∃ (n : ℕ), row_elements n = 101 ∧ third_number n = 4950 :=
by sorry

end pascal_triangle_101_row_third_number_l858_85803


namespace focus_on_negative_y_axis_l858_85897

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 + y = 0

-- Define the focus of a parabola
def focus (p q : ℝ) : Prop := ∃ (a : ℝ), p = 0 ∧ q = -1/(4*a)

-- Theorem statement
theorem focus_on_negative_y_axis :
  ∃ (p q : ℝ), focus p q ∧ ∀ (x y : ℝ), parabola x y → q < 0 :=
sorry

end focus_on_negative_y_axis_l858_85897


namespace total_marbles_l858_85879

/-- Given 5 bags with 5 marbles each and 1 bag with 8 marbles,
    the total number of marbles in all 6 bags is 33. -/
theorem total_marbles (bags_of_five : Nat) (marbles_per_bag : Nat) (extra_bag : Nat) :
  bags_of_five = 5 →
  marbles_per_bag = 5 →
  extra_bag = 8 →
  bags_of_five * marbles_per_bag + extra_bag = 33 := by
  sorry

end total_marbles_l858_85879


namespace arithmetic_expression_equality_l858_85862

theorem arithmetic_expression_equality : 70 + 5 * 12 / (180 / 3) = 71 := by
  sorry

end arithmetic_expression_equality_l858_85862


namespace imaginary_part_of_complex_fraction_l858_85890

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (2 - I) / I
  (z.im : ℝ) = -2 := by sorry

end imaginary_part_of_complex_fraction_l858_85890


namespace f_monotonically_decreasing_l858_85826

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

-- Theorem statement
theorem f_monotonically_decreasing :
  ∀ x y, -2 < x ∧ x < y ∧ y < 1 → f x > f y := by
  sorry

end f_monotonically_decreasing_l858_85826


namespace button_ratio_problem_l858_85845

/-- Represents the button problem with Mark, Shane, and Sam -/
theorem button_ratio_problem (initial_buttons : ℕ) (shane_multiplier : ℕ) (final_buttons : ℕ) :
  initial_buttons = 14 →
  shane_multiplier = 3 →
  final_buttons = 28 →
  let total_after_shane := initial_buttons + shane_multiplier * initial_buttons
  let sam_took := total_after_shane - final_buttons
  (sam_took : ℚ) / total_after_shane = 1 / 2 := by sorry

end button_ratio_problem_l858_85845


namespace fair_ride_cost_l858_85886

theorem fair_ride_cost (total_tickets : ℕ) (booth_tickets : ℕ) (num_rides : ℕ) 
  (h1 : total_tickets = 79) 
  (h2 : booth_tickets = 23) 
  (h3 : num_rides = 8) : 
  (total_tickets - booth_tickets) / num_rides = 7 := by
  sorry

end fair_ride_cost_l858_85886


namespace modulus_of_specific_complex_number_l858_85866

open Complex

theorem modulus_of_specific_complex_number :
  let z : ℂ := (2 - I) / (2 + I)
  ‖z‖ = 1 := by
  sorry

end modulus_of_specific_complex_number_l858_85866


namespace y₁_gt_y₂_l858_85853

/-- A linear function y = -2x + 3 --/
def f (x : ℝ) : ℝ := -2 * x + 3

/-- Point P₁ on the graph of f --/
def P₁ : ℝ × ℝ := (-2, f (-2))

/-- Point P₂ on the graph of f --/
def P₂ : ℝ × ℝ := (3, f 3)

/-- The y-coordinate of P₁ --/
def y₁ : ℝ := P₁.2

/-- The y-coordinate of P₂ --/
def y₂ : ℝ := P₂.2

theorem y₁_gt_y₂ : y₁ > y₂ := by
  sorry

end y₁_gt_y₂_l858_85853


namespace f_decreasing_interval_l858_85834

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x + 4

-- State the theorem
theorem f_decreasing_interval :
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 6 ∧ f x₂ = 2) →  -- Maximum and minimum conditions
  (∀ x, f x ≤ 6) →                           -- 6 is the maximum value
  (∀ x, f x ≥ 2) →                           -- 2 is the minimum value
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f x > f y) :=
by sorry

end f_decreasing_interval_l858_85834


namespace gcd_lcm_18_24_l858_85812

theorem gcd_lcm_18_24 :
  (Nat.gcd 18 24 = 6) ∧ (Nat.lcm 18 24 = 72) := by
  sorry

end gcd_lcm_18_24_l858_85812


namespace hyperbola_properties_l858_85858

/-- A hyperbola with foci on the x-axis, real axis length 4√2, and eccentricity √5/2 -/
structure Hyperbola where
  /-- Real axis length -/
  real_axis_length : ℝ
  real_axis_length_eq : real_axis_length = 4 * Real.sqrt 2
  /-- Eccentricity -/
  e : ℝ
  e_eq : e = Real.sqrt 5 / 2

/-- Standard form of the hyperbola equation -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 2 = 1

/-- Equation of the trajectory of point Q -/
def trajectory_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 4 = 1 ∧ x ≠ 2 * Real.sqrt 2 ∧ x ≠ -2 * Real.sqrt 2

theorem hyperbola_properties (h : Hyperbola) :
  (∀ x y, standard_equation h x y ↔ 
    x^2 / (2 * Real.sqrt 2)^2 - y^2 / ((Real.sqrt 5 / 2) * 2 * Real.sqrt 2)^2 = 1) ∧
  (∀ x y, trajectory_equation h x y ↔
    x^2 / 8 - y^2 / 4 = 1 ∧ x ≠ 2 * Real.sqrt 2 ∧ x ≠ -2 * Real.sqrt 2) :=
sorry

end hyperbola_properties_l858_85858


namespace product_of_roots_eq_one_l858_85837

theorem product_of_roots_eq_one :
  let f : ℝ → ℝ := λ x => x^(Real.log x / Real.log 5) - 25
  ∃ (r₁ r₂ : ℝ), (f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ ≠ r₂) ∧ r₁ * r₂ = 1 :=
by sorry

end product_of_roots_eq_one_l858_85837


namespace polynomial_value_l858_85835

theorem polynomial_value (a b : ℝ) (h : |a - 2| + (b + 1/2)^2 = 0) :
  (2*a*b^2 + a^2*b) - (3*a*b^2 + a^2*b - 1) = 1/2 := by
  sorry

end polynomial_value_l858_85835


namespace h_transformation_l858_85807

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the transformation h
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := 2 * f x + 3

-- Theorem statement
theorem h_transformation (f : ℝ → ℝ) (x : ℝ) : 
  h f x = 2 * f x + 3 := by
  sorry

end h_transformation_l858_85807


namespace complex_number_in_third_quadrant_l858_85824

theorem complex_number_in_third_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := -5 * i / (2 + 3 * i)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_third_quadrant_l858_85824


namespace power_relation_l858_85831

theorem power_relation (a : ℝ) (b : ℝ) (h : a ^ b = 1 / 8) : a ^ (-3 * b) = 512 := by
  sorry

end power_relation_l858_85831


namespace petri_dishes_count_l858_85832

/-- The number of petri dishes in the biology lab -/
def num_petri_dishes : ℕ :=
  10800

/-- The total number of germs in the lab -/
def total_germs : ℕ :=
  5400000

/-- The number of germs in a single dish -/
def germs_per_dish : ℕ :=
  500

/-- Theorem stating that the number of petri dishes is correct -/
theorem petri_dishes_count :
  num_petri_dishes = total_germs / germs_per_dish :=
by sorry

end petri_dishes_count_l858_85832


namespace dilan_initial_marbles_l858_85842

/-- The number of people involved in the marble redistribution --/
def num_people : ℕ := 4

/-- The number of marbles each person has after redistribution --/
def marbles_after : ℕ := 15

/-- Martha's initial number of marbles --/
def martha_initial : ℕ := 20

/-- Phillip's initial number of marbles --/
def phillip_initial : ℕ := 19

/-- Veronica's initial number of marbles --/
def veronica_initial : ℕ := 7

/-- The theorem stating Dilan's initial number of marbles --/
theorem dilan_initial_marbles :
  (num_people * marbles_after) - (martha_initial + phillip_initial + veronica_initial) = 14 :=
by sorry

end dilan_initial_marbles_l858_85842


namespace max_two_digit_composite_relatively_prime_l858_85817

/-- A number is two-digit if it's between 10 and 99 inclusive -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A number is composite if it has a factor other than 1 and itself -/
def isComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- Two numbers are relatively prime if their greatest common divisor is 1 -/
def areRelativelyPrime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The set of numbers satisfying our conditions -/
def validSet (S : Finset ℕ) : Prop :=
  ∀ n ∈ S, isTwoDigit n ∧ isComposite n ∧
  ∀ m ∈ S, m ≠ n → areRelativelyPrime m n

theorem max_two_digit_composite_relatively_prime :
  (∃ S : Finset ℕ, validSet S ∧ S.card = 4) ∧
  ∀ T : Finset ℕ, validSet T → T.card ≤ 4 := by
  sorry

end max_two_digit_composite_relatively_prime_l858_85817


namespace q_investment_time_l858_85887

-- Define the investment ratio
def investment_ratio : ℚ := 7 / 5

-- Define the profit ratio
def profit_ratio : ℚ := 7 / 10

-- Define P's investment time in months
def p_time : ℚ := 2

-- Define Q's investment time as a variable
variable (q_time : ℚ)

-- Theorem statement
theorem q_investment_time : 
  (investment_ratio * p_time) / (q_time / investment_ratio) = profit_ratio → q_time = 4 := by
  sorry

end q_investment_time_l858_85887


namespace even_quadratic_implies_m_eq_two_l858_85885

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + (m-2)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-2)*x + 1

theorem even_quadratic_implies_m_eq_two (m : ℝ) (h : IsEven (f m)) : m = 2 := by
  sorry

end even_quadratic_implies_m_eq_two_l858_85885


namespace f_composed_with_g_l858_85804

def f (x : ℝ) : ℝ := 3 * x - 4

def g (x : ℝ) : ℝ := x + 2

theorem f_composed_with_g : f (2 + g 3) = 17 := by
  sorry

end f_composed_with_g_l858_85804


namespace quadratic_roots_property_l858_85846

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 14 = 0) → 
  (3 * q^2 - 5 * q - 14 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
sorry

end quadratic_roots_property_l858_85846


namespace prob_odd_females_committee_l858_85813

/-- The number of men in the pool of candidates -/
def num_men : ℕ := 5

/-- The number of women in the pool of candidates -/
def num_women : ℕ := 4

/-- The size of the committee to be formed -/
def committee_size : ℕ := 3

/-- The probability of selecting a committee with an odd number of female members -/
def prob_odd_females : ℚ := 11 / 21

/-- Theorem stating that the probability of selecting a committee of three members
    with an odd number of female members from a pool of five men and four women,
    where all candidates are equally likely to be chosen, is 11/21 -/
theorem prob_odd_females_committee :
  let total_candidates := num_men + num_women
  let total_committees := Nat.choose total_candidates committee_size
  let committees_one_female := Nat.choose num_women 1 * Nat.choose num_men 2
  let committees_three_females := Nat.choose num_women 3 * Nat.choose num_men 0
  let favorable_outcomes := committees_one_female + committees_three_females
  (favorable_outcomes : ℚ) / total_committees = prob_odd_females := by
  sorry


end prob_odd_females_committee_l858_85813


namespace total_cost_with_tax_l858_85821

def nike_price : ℝ := 150
def boots_price : ℝ := 120
def tax_rate : ℝ := 0.1

theorem total_cost_with_tax :
  let pre_tax_total := nike_price + boots_price
  let tax_amount := pre_tax_total * tax_rate
  let total_with_tax := pre_tax_total + tax_amount
  total_with_tax = 297 := by sorry

end total_cost_with_tax_l858_85821


namespace tenth_term_of_specific_geometric_sequence_l858_85802

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The tenth term of a geometric sequence with first term 5 and common ratio 3/4 -/
theorem tenth_term_of_specific_geometric_sequence :
  geometric_sequence 5 (3/4) 10 = 98415/262144 := by
  sorry

end tenth_term_of_specific_geometric_sequence_l858_85802


namespace age_sum_proof_l858_85899

/-- Given the age relationship between Michael and Emily, prove that the sum of their current ages is 32. -/
theorem age_sum_proof (M E : ℚ) : 
  M = E + 9 ∧ 
  M + 5 = 3 * (E - 3) → 
  M + E = 32 := by
  sorry

end age_sum_proof_l858_85899


namespace polynomial_divisibility_l858_85852

theorem polynomial_divisibility : ∃ (q : ℝ → ℝ), ∀ x : ℝ, 
  4 * x^2 - 6 * x - 18 = (x - 3) * q x := by
  sorry

end polynomial_divisibility_l858_85852


namespace circle_plus_four_two_l858_85867

/-- Definition of the ⊕ operation -/
def circle_plus (a b : ℝ) : ℝ := 5 * a + 6 * b

/-- Theorem stating that 4 ⊕ 2 = 32 -/
theorem circle_plus_four_two : circle_plus 4 2 = 32 := by
  sorry

end circle_plus_four_two_l858_85867


namespace inequality_chain_l858_85829

theorem inequality_chain (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end inequality_chain_l858_85829


namespace violet_marbles_indeterminate_l858_85820

/-- Represents the number of marbles Dan has -/
structure DansMarbles where
  initialGreen : ℝ
  takenGreen : ℝ
  finalGreen : ℝ
  violet : ℝ

/-- Theorem stating that the number of violet marbles cannot be determined -/
theorem violet_marbles_indeterminate (d : DansMarbles) 
  (h1 : d.initialGreen = 32)
  (h2 : d.takenGreen = 23)
  (h3 : d.finalGreen = 9)
  (h4 : d.initialGreen - d.takenGreen = d.finalGreen) :
  ∀ v : ℝ, ∃ d' : DansMarbles, d'.initialGreen = d.initialGreen ∧ 
                                d'.takenGreen = d.takenGreen ∧ 
                                d'.finalGreen = d.finalGreen ∧ 
                                d'.violet = v :=
sorry

end violet_marbles_indeterminate_l858_85820


namespace imaginary_part_of_complex_fraction_l858_85861

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 * i + 1) / (1 - i)
  Complex.im z = 2 := by
  sorry

end imaginary_part_of_complex_fraction_l858_85861


namespace complex_magnitude_l858_85844

theorem complex_magnitude (z : ℂ) (h : z / (2 - Complex.I) = 2 * Complex.I) : 
  Complex.abs (z + 1) = Real.sqrt 17 := by
sorry

end complex_magnitude_l858_85844


namespace equation_representation_l858_85874

theorem equation_representation (x : ℝ) : 
  (2 * x + 4 = 8) → (∃ y : ℝ, y = 2 * x + 4 ∧ y = 8) := by
  sorry

end equation_representation_l858_85874


namespace hex_palindrome_probability_l858_85815

/-- Represents a hexadecimal digit (0-15) -/
def HexDigit := Fin 16

/-- Represents a 6-digit hexadecimal palindrome -/
structure HexPalindrome where
  a : HexDigit
  b : HexDigit
  c : HexDigit
  value : ℕ := 1048592 * a.val + 65792 * b.val + 4096 * c.val

/-- Predicate to check if a natural number is a hexadecimal palindrome -/
def isHexPalindrome (n : ℕ) : Prop := sorry

/-- The total number of 6-digit hexadecimal palindromes -/
def totalPalindromes : ℕ := 3840

/-- The number of 6-digit hexadecimal palindromes that, when divided by 17, 
    result in another hexadecimal palindrome -/
def validPalindromes : ℕ := sorry

theorem hex_palindrome_probability : 
  (validPalindromes : ℚ) / totalPalindromes = 1 / 15 := by sorry

end hex_palindrome_probability_l858_85815


namespace bake_sale_brownie_cost_l858_85864

/-- Proves that the cost per brownie is $2 given the conditions of the bake sale --/
theorem bake_sale_brownie_cost (total_revenue : ℝ) (num_pans : ℕ) (pieces_per_pan : ℕ) :
  total_revenue = 32 →
  num_pans = 2 →
  pieces_per_pan = 8 →
  (total_revenue / (num_pans * pieces_per_pan : ℝ)) = 2 := by
  sorry

end bake_sale_brownie_cost_l858_85864


namespace percentage_calculation_l858_85827

theorem percentage_calculation (N P : ℝ) (h1 : N = 75) (h2 : N = (P / 100) * N + 63) : P = 16 := by
  sorry

end percentage_calculation_l858_85827


namespace intersection_of_sets_l858_85883

theorem intersection_of_sets (M N : Set ℝ) : 
  M = {x : ℝ | Real.sqrt (x + 1) ≥ 0} →
  N = {x : ℝ | x^2 + x - 2 < 0} →
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_sets_l858_85883


namespace negative_six_times_negative_one_l858_85859

theorem negative_six_times_negative_one : (-6 : ℤ) * (-1 : ℤ) = 6 := by sorry

end negative_six_times_negative_one_l858_85859


namespace sum_10_terms_formula_l858_85860

/-- An arithmetic progression with sum of 4th and 12th terms equal to 20 -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_4_12 : a + 3*d + a + 11*d = 20

/-- The sum of the first 10 terms of the arithmetic progression -/
def sum_10_terms (ap : ArithmeticProgression) : ℝ :=
  5 * (2*ap.a + 9*ap.d)

/-- Theorem: The sum of the first 10 terms equals 100 - 25d -/
theorem sum_10_terms_formula (ap : ArithmeticProgression) :
  sum_10_terms ap = 100 - 25*ap.d := by
  sorry

end sum_10_terms_formula_l858_85860


namespace towel_shrinkage_l858_85830

theorem towel_shrinkage (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let original_area := L * B
  let shrunk_length := 0.8 * L
  let shrunk_breadth := 0.9 * B
  let shrunk_area := shrunk_length * shrunk_breadth
  let cumulative_shrunk_area := 0.95 * shrunk_area
  let folded_area := 0.5 * cumulative_shrunk_area
  let percentage_change := (folded_area - original_area) / original_area * 100
  percentage_change = -65.8 := by
sorry

end towel_shrinkage_l858_85830


namespace floor_of_4_7_l858_85888

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end floor_of_4_7_l858_85888


namespace equation_solution_l858_85818

theorem equation_solution :
  ∀ x : ℚ, (x ≠ 4 ∧ x ≠ -6) →
  ((x + 10) / (x - 4) = (x - 3) / (x + 6)) →
  x = -48 / 23 :=
by
  sorry

end equation_solution_l858_85818


namespace triangle_angle_C_l858_85810

theorem triangle_angle_C (A B C : Real) (a b c : Real) :
  a + b + c = Real.sqrt 2 + 1 →
  (1/2) * a * b * Real.sin C = (1/6) * Real.sin C →
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →
  C = π / 3 := by
  sorry

end triangle_angle_C_l858_85810


namespace card_game_combinations_l858_85868

theorem card_game_combinations : Nat.choose 52 13 = 635013587600 := by
  sorry

end card_game_combinations_l858_85868


namespace only_negative_four_squared_is_correct_l858_85843

theorem only_negative_four_squared_is_correct : 
  (2^4 ≠ 8) ∧ 
  (-4^2 = -16) ∧ 
  (-8 - 8 ≠ 0) ∧ 
  ((-3)^2 ≠ 6) := by
  sorry

end only_negative_four_squared_is_correct_l858_85843


namespace product_of_roots_quadratic_l858_85893

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - x₁ - 6 = 0) → (x₂^2 - x₂ - 6 = 0) → x₁ * x₂ = -6 := by
  sorry

end product_of_roots_quadratic_l858_85893


namespace kids_difference_l858_85836

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 22) 
  (h2 : tuesday = 14) : 
  monday - tuesday = 8 := by
sorry

end kids_difference_l858_85836


namespace tangent_line_at_one_l858_85828

def f (x : ℝ) : ℝ := x^4 - 2*x^3

theorem tangent_line_at_one (x y : ℝ) :
  (y - f 1 = (4 - 6) * (x - 1)) ↔ (y = -2*x + 1) := by sorry

end tangent_line_at_one_l858_85828


namespace cubic_integer_roots_imply_b_form_l858_85800

theorem cubic_integer_roots_imply_b_form (a b : ℤ) 
  (h : ∃ (u v w : ℤ), u^3 - a*u^2 - b = 0 ∧ v^3 - a*v^2 - b = 0 ∧ w^3 - a*w^2 - b = 0) :
  ∃ (d k : ℤ), b = d * k^2 ∧ ∃ (m : ℤ), a = d * m :=
by sorry

end cubic_integer_roots_imply_b_form_l858_85800


namespace evaluate_expression_l858_85873

theorem evaluate_expression : 3 + 3 * (3 ^ (3 ^ 3)) - 3 ^ 3 = 22876792454937 := by
  sorry

end evaluate_expression_l858_85873


namespace square_area_from_vertices_l858_85884

/-- The area of a square with adjacent vertices at (1, 3) and (5, -1) is 32 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (5, -1)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 32 := by
  sorry

end square_area_from_vertices_l858_85884


namespace greatest_n_for_given_conditions_l858_85822

theorem greatest_n_for_given_conditions (x : ℤ) (N : ℝ) : 
  (N * 10^x < 210000 ∧ x ≤ 4) → 
  ∃ (max_N : ℤ), max_N = 20 ∧ ∀ (m : ℤ), (m : ℝ) * 10^4 < 210000 → m ≤ max_N :=
sorry

end greatest_n_for_given_conditions_l858_85822


namespace f_2x_l858_85869

/-- Given a function f(x) = x^2 - 1, prove that f(2x) = 4x^2 - 1 --/
theorem f_2x (x : ℝ) : (fun x => x^2 - 1) (2*x) = 4*x^2 - 1 := by
  sorry

end f_2x_l858_85869


namespace arithmetic_sequence_sum_l858_85875

/-- Given an arithmetic sequence with first term 3 and last term 27,
    the sum of the two terms immediately preceding 27 is 42. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 0 = 3 →  -- first term is 3
  (∃ k : ℕ, a k = 27 ∧ ∀ n > k, a n ≠ 27) →  -- 27 is the last term
  (∃ m : ℕ, a (m - 1) + a m = 42 ∧ a (m + 1) = 27) :=
by sorry

end arithmetic_sequence_sum_l858_85875


namespace robot_sorting_problem_l858_85808

/-- Represents the sorting capacity of robots -/
structure RobotSorting where
  typeA : ℕ  -- Number of type A robots
  typeB : ℕ  -- Number of type B robots
  totalPackages : ℕ  -- Total packages sorted per hour

/-- Theorem representing the robot sorting problem -/
theorem robot_sorting_problem 
  (scenario1 : RobotSorting)
  (scenario2 : RobotSorting)
  (h1 : scenario1.typeA = 80 ∧ scenario1.typeB = 100 ∧ scenario1.totalPackages = 8200)
  (h2 : scenario2.typeA = 50 ∧ scenario2.typeB = 50 ∧ scenario2.totalPackages = 4500)
  (totalNewRobots : ℕ)
  (h3 : totalNewRobots = 200)
  (minNewPackages : ℕ)
  (h4 : minNewPackages = 9000) :
  ∃ (maxTypeA : ℕ),
    maxTypeA ≤ totalNewRobots ∧
    ∀ (newTypeA : ℕ),
      newTypeA ≤ totalNewRobots →
      (40 * newTypeA + 50 * (totalNewRobots - newTypeA) ≥ minNewPackages →
       newTypeA ≤ maxTypeA) ∧
    40 * maxTypeA + 50 * (totalNewRobots - maxTypeA) ≥ minNewPackages ∧
    maxTypeA = 100 :=
sorry

end robot_sorting_problem_l858_85808


namespace polygon_with_170_diagonals_has_20_sides_l858_85863

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 170 diagonals has 20 sides -/
theorem polygon_with_170_diagonals_has_20_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 170 → n = 20 :=
by
  sorry

#check polygon_with_170_diagonals_has_20_sides

end polygon_with_170_diagonals_has_20_sides_l858_85863


namespace A_3_2_l858_85889

def A : Nat → Nat → Nat
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2 : A 3 2 = 29 := by sorry

end A_3_2_l858_85889


namespace hcd_7350_165_minus_15_l858_85882

theorem hcd_7350_165_minus_15 : Nat.gcd 7350 165 - 15 = 0 := by
  sorry

end hcd_7350_165_minus_15_l858_85882


namespace nancy_shoe_count_l858_85814

def shoe_count (boots slippers heels : ℕ) : ℕ :=
  2 * (boots + slippers + heels)

theorem nancy_shoe_count :
  ∀ (boots slippers heels : ℕ),
    boots = 6 →
    slippers = boots + 9 →
    heels = 3 * (boots + slippers) →
    shoe_count boots slippers heels = 168 := by
  sorry

end nancy_shoe_count_l858_85814


namespace reciprocal_multiple_l858_85850

theorem reciprocal_multiple (x : ℝ) (k : ℝ) (h1 : x > 0) (h2 : x = 8) (h3 : x + 8 = k * (1 / x)) : k = 128 := by
  sorry

end reciprocal_multiple_l858_85850


namespace keith_digimon_packs_l858_85881

/-- The cost of one pack of Digimon cards in dollars -/
def digimon_pack_cost : ℚ := 445/100

/-- The cost of a deck of baseball cards in dollars -/
def baseball_deck_cost : ℚ := 606/100

/-- The total amount Keith spent on cards in dollars -/
def total_spent : ℚ := 2386/100

/-- The number of Digimon card packs Keith bought -/
def digimon_packs : ℕ := 4

theorem keith_digimon_packs :
  digimon_packs * digimon_pack_cost + baseball_deck_cost = total_spent :=
sorry

end keith_digimon_packs_l858_85881


namespace return_trip_amount_l858_85870

def initial_amount : ℝ := 50
def gasoline_cost : ℝ := 8
def lunch_cost : ℝ := 15.65
def gift_cost_per_person : ℝ := 5
def number_of_people : ℕ := 2
def grandma_gift_per_person : ℝ := 10

def total_expenses : ℝ := gasoline_cost + lunch_cost + (gift_cost_per_person * number_of_people)
def remaining_after_expenses : ℝ := initial_amount - total_expenses
def total_grandma_gift : ℝ := grandma_gift_per_person * number_of_people
def final_amount : ℝ := remaining_after_expenses + total_grandma_gift

theorem return_trip_amount :
  final_amount = 36.35 := by sorry

end return_trip_amount_l858_85870


namespace count_counterexamples_l858_85823

def sum_of_digits (n : ℕ) : ℕ := sorry

def has_no_zero_digit (n : ℕ) : Prop := sorry

def counterexample (n : ℕ) : Prop :=
  sum_of_digits n = 5 ∧ has_no_zero_digit n ∧ ¬ Nat.Prime n

theorem count_counterexamples : 
  ∃ (S : Finset ℕ), S.card = 6 ∧ ∀ n, n ∈ S ↔ counterexample n :=
sorry

end count_counterexamples_l858_85823


namespace ladder_distance_l858_85839

theorem ladder_distance (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end ladder_distance_l858_85839


namespace shirt_price_l858_85849

/-- The cost of one pair of jeans in dollars -/
def jean_cost : ℝ := sorry

/-- The cost of one shirt in dollars -/
def shirt_cost : ℝ := sorry

/-- First condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jean_cost + 2 * shirt_cost = 69

/-- Second condition: 2 pairs of jeans and 3 shirts cost $81 -/
axiom condition2 : 2 * jean_cost + 3 * shirt_cost = 81

/-- Theorem: The cost of one shirt is $21 -/
theorem shirt_price : shirt_cost = 21 := by sorry

end shirt_price_l858_85849


namespace percentage_loss_l858_85857

def cost_price : ℝ := 1800
def selling_price : ℝ := 1350

theorem percentage_loss : (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end percentage_loss_l858_85857


namespace complex_simplification_l858_85851

theorem complex_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  7 * (2 - 2*i) + 4*i * (7 - 3*i) = 26 + 14*i :=
by
  sorry

end complex_simplification_l858_85851


namespace base_4_arithmetic_l858_85805

/-- Converts a base 4 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def to_base_4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_4_arithmetic :
  to_base_4 (to_base_10 [0, 3, 2] - to_base_10 [1, 0, 1] + to_base_10 [2, 2, 3]) = [1, 1, 1, 1] := by
  sorry

end base_4_arithmetic_l858_85805


namespace tower_surface_area_l858_85811

/-- Represents a layer in the tower -/
structure Layer where
  cubes : ℕ
  exposed_top : ℕ
  exposed_sides : ℕ

/-- Represents the tower of cubes -/
def Tower : List Layer := [
  { cubes := 1, exposed_top := 1, exposed_sides := 5 },
  { cubes := 3, exposed_top := 3, exposed_sides := 8 },
  { cubes := 4, exposed_top := 4, exposed_sides := 6 },
  { cubes := 6, exposed_top := 6, exposed_sides := 0 }
]

/-- The total number of cubes in the tower -/
def total_cubes : ℕ := (Tower.map (·.cubes)).sum

/-- The exposed surface area of the tower -/
def exposed_surface_area : ℕ := 
  (Tower.map (·.exposed_top)).sum + (Tower.map (·.exposed_sides)).sum

theorem tower_surface_area : 
  total_cubes = 14 ∧ exposed_surface_area = 29 := by
  sorry

end tower_surface_area_l858_85811


namespace no_real_solution_system_l858_85878

theorem no_real_solution_system :
  ¬∃ (x y z : ℝ), (x + y + 2 + 4*x*y = 0) ∧ 
                  (y + z + 2 + 4*y*z = 0) ∧ 
                  (z + x + 2 + 4*z*x = 0) := by
  sorry

end no_real_solution_system_l858_85878


namespace find_y_value_l858_85838

theorem find_y_value (a b x y : ℤ) : 
  (a + b + 100 + 200300 + x) / 5 = 250 →
  (a + b + 300 + 150100 + x + y) / 6 = 200 →
  a % 5 = 0 →
  b % 5 = 0 →
  y = 49800 := by
sorry

end find_y_value_l858_85838


namespace lot_length_l858_85841

/-- Given a rectangular lot with width 20 meters, height 2 meters, and volume 1600 cubic meters,
    prove that the length of the lot is 40 meters. -/
theorem lot_length (width : ℝ) (height : ℝ) (volume : ℝ) (length : ℝ) :
  width = 20 →
  height = 2 →
  volume = 1600 →
  volume = length * width * height →
  length = 40 := by
  sorry

end lot_length_l858_85841


namespace fish_problem_solution_l858_85865

/-- Calculates the number of fish added on day 7 given the initial conditions and daily changes --/
def fish_added_day_7 (initial : ℕ) (double : ℕ → ℕ) (remove_third : ℕ → ℕ) (remove_fourth : ℕ → ℕ) (final : ℕ) : ℕ :=
  let day1 := initial
  let day2 := double day1
  let day3 := remove_third (double day2)
  let day4 := double day3
  let day5 := remove_fourth (double day4)
  let day6 := double day5
  let day7_before_adding := double day6
  final - day7_before_adding

theorem fish_problem_solution :
  fish_added_day_7 6 (λ x => 2 * x) (λ x => x - x / 3) (λ x => x - x / 4) 207 = 15 := by
  sorry

#eval fish_added_day_7 6 (λ x => 2 * x) (λ x => x - x / 3) (λ x => x - x / 4) 207

end fish_problem_solution_l858_85865


namespace cubic_root_sum_ninth_power_l858_85809

theorem cubic_root_sum_ninth_power (u v w : ℂ) : 
  (u^3 - 3*u - 1 = 0) → 
  (v^3 - 3*v - 1 = 0) → 
  (w^3 - 3*w - 1 = 0) → 
  u^9 + v^9 + w^9 = 246 := by sorry

end cubic_root_sum_ninth_power_l858_85809


namespace spherical_segment_surface_area_equals_circle_area_l858_85891

/-- Given a spherical segment with radius R and height H, and a circle with radius b
    where b² = 2RH, the surface area of the spherical segment (2πRH) is equal to
    the area of the circle (πb²). -/
theorem spherical_segment_surface_area_equals_circle_area
  (R H b : ℝ) (h : b^2 = 2 * R * H) :
  2 * Real.pi * R * H = Real.pi * b^2 := by
  sorry

end spherical_segment_surface_area_equals_circle_area_l858_85891


namespace air_quality_probability_l858_85856

theorem air_quality_probability (p_good : ℝ) (p_consecutive : ℝ) :
  p_good = 0.8 →
  p_consecutive = 0.6 →
  p_good * (p_consecutive / p_good) = 0.75 :=
by sorry

end air_quality_probability_l858_85856


namespace prime_sum_equality_l858_85898

theorem prime_sum_equality (p q n : ℕ) : 
  Prime p → Prime q → 0 < n → 
  p * (p + 3) + q * (q + 3) = n * (n + 3) → 
  ((p = 2 ∧ q = 3 ∧ n = 4) ∨ 
   (p = 3 ∧ q = 2 ∧ n = 4) ∨ 
   (p = 3 ∧ q = 7 ∧ n = 8) ∨ 
   (p = 7 ∧ q = 3 ∧ n = 8)) := by
sorry

end prime_sum_equality_l858_85898


namespace rainy_days_count_l858_85825

theorem rainy_days_count (n : ℕ) : 
  (∃ (rainy_days non_rainy_days : ℕ),
    rainy_days + non_rainy_days = 7 ∧
    n * rainy_days + 5 * non_rainy_days = 22 ∧
    5 * non_rainy_days - n * rainy_days = 8) →
  (∃ (rainy_days : ℕ), rainy_days = 4) :=
by sorry

end rainy_days_count_l858_85825


namespace scientific_notation_of_1268000000_l858_85819

theorem scientific_notation_of_1268000000 :
  (1268000000 : ℝ) = 1.268 * (10 ^ 9) := by sorry

end scientific_notation_of_1268000000_l858_85819


namespace market_spending_l858_85892

theorem market_spending (mildred_spent candice_spent joseph_spent_percentage joseph_spent mom_total : ℝ) :
  mildred_spent = 25 →
  candice_spent = 35 →
  joseph_spent_percentage = 0.8 →
  joseph_spent = 45 →
  mom_total = 150 →
  mom_total - (mildred_spent + candice_spent + joseph_spent) = 45 :=
by
  sorry

end market_spending_l858_85892


namespace min_value_xy_minus_2x_l858_85895

theorem min_value_xy_minus_2x (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y * Real.log x + y * Real.log y = Real.exp x) : 
  ∃ (m : ℝ), m = 2 - 2 * Real.log 2 ∧ 
  ∀ (z : ℝ), z > 0 → y * z * Real.log (y * z) = z * Real.exp z → 
  x * y - 2 * x ≥ m := by
  sorry

end min_value_xy_minus_2x_l858_85895


namespace noelle_walking_speed_l858_85847

/-- Noelle's walking problem -/
theorem noelle_walking_speed (d : ℝ) (h : d > 0) : 
  let v : ℝ := (2 * d) / (d / 15 + d / 5)
  v = 3 := by sorry

end noelle_walking_speed_l858_85847


namespace circles_common_chord_and_diameter_l858_85848

-- Define the two circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the circle with common chord as diameter
def circle_with_common_chord_diameter (x y : ℝ) : Prop := 
  (x + 8/5)^2 + (y - 6/5)^2 = 36/5

-- Theorem statement
theorem circles_common_chord_and_diameter :
  (∃ x y : ℝ, C1 x y ∧ C2 x y ∧ common_chord x y) →
  (∃ a b : ℝ, common_chord a b ∧ 
    (a - (-4))^2 + (b - 0)^2 = 5) ∧
  (∀ x y : ℝ, circle_with_common_chord_diameter x y ↔
    (∃ t : ℝ, x = -4 * (1 - t) + 4/5 * t ∧ 
              y = 0 * (1 - t) + 12/5 * t ∧ 
              0 ≤ t ∧ t ≤ 1)) := by
  sorry

end circles_common_chord_and_diameter_l858_85848


namespace negation_of_universal_proposition_l858_85877

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 4 * x^2 - 3 * x + 2 < 0) ↔ (∃ x : ℝ, 4 * x^2 - 3 * x + 2 ≥ 0) := by
  sorry

end negation_of_universal_proposition_l858_85877


namespace pencil_count_l858_85855

theorem pencil_count (group_size : ℕ) (num_groups : ℕ) (h1 : group_size = 11) (h2 : num_groups = 14) :
  group_size * num_groups = 154 := by
  sorry

end pencil_count_l858_85855
