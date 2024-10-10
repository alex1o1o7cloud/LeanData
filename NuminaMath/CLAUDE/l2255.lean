import Mathlib

namespace translation_problem_l2255_225599

/-- A translation in the complex plane -/
def ComplexTranslation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

theorem translation_problem (T : ℂ → ℂ) (h : T = ComplexTranslation (3 + 5*I)) :
  T (3 - I) = 6 + 4*I := by
  sorry

end translation_problem_l2255_225599


namespace license_plate_count_l2255_225559

/-- The number of consonants in the English alphabet (including Y) -/
def num_consonants : ℕ := 21

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of digits (0 through 9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_plates : ℕ := num_consonants * num_consonants * num_vowels * num_vowels * num_digits

theorem license_plate_count : total_plates = 110250 := by sorry

end license_plate_count_l2255_225559


namespace special_numbers_theorem_l2255_225585

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n > 0 ∧ n - sum_of_digits n = 2007

theorem special_numbers_theorem : 
  {n : ℕ | satisfies_condition n} = {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019} :=
by sorry

end special_numbers_theorem_l2255_225585


namespace restaurant_combinations_l2255_225519

theorem restaurant_combinations (menu_items : ℕ) (special_dish : ℕ) : menu_items = 12 ∧ special_dish = 1 →
  (menu_items - special_dish) * (menu_items - special_dish) + 
  2 * special_dish * (menu_items - special_dish) = 143 := by
  sorry

end restaurant_combinations_l2255_225519


namespace log_expression_equals_one_l2255_225583

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_one :
  (log10 5)^2 + log10 50 * log10 2 = 1 := by sorry

end log_expression_equals_one_l2255_225583


namespace max_sum_at_9_l2255_225598

/-- An arithmetic sequence with first term 1 and common difference d -/
def arithmetic_sequence (d : ℚ) : ℕ → ℚ := λ n => 1 + (n - 1 : ℚ) * d

/-- The sum of the first n terms of the arithmetic sequence -/
def S (d : ℚ) (n : ℕ) : ℚ := (n : ℚ) * (2 + (n - 1 : ℚ) * d) / 2

/-- The theorem stating that Sn reaches its maximum when n = 9 -/
theorem max_sum_at_9 (d : ℚ) (h : -2/17 < d ∧ d < -1/9) :
  ∀ k : ℕ, S d 9 ≥ S d k :=
sorry

end max_sum_at_9_l2255_225598


namespace highlight_film_average_time_l2255_225586

/-- The average time each player gets in the highlight film -/
def average_time (durations : List Nat) : Rat :=
  (durations.sum / 60) / durations.length

/-- Theorem: Given the video durations for 5 players, the average time each player gets is 2 minutes -/
theorem highlight_film_average_time :
  let durations := [130, 145, 85, 60, 180]
  average_time durations = 2 := by sorry

end highlight_film_average_time_l2255_225586


namespace square_plus_reciprocal_square_l2255_225568

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end square_plus_reciprocal_square_l2255_225568


namespace trapezium_side_length_l2255_225580

theorem trapezium_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 15 → area = 285 → area = (a + b) * h / 2 → b = 18 :=
by sorry

end trapezium_side_length_l2255_225580


namespace divisibility_property_l2255_225501

theorem divisibility_property (n : ℕ) (a b : ℤ) :
  (a ≠ b) →
  (∀ m : ℕ, (n^m : ℤ) ∣ (a^m - b^m)) →
  (n : ℤ) ∣ a ∧ (n : ℤ) ∣ b :=
by sorry

end divisibility_property_l2255_225501


namespace derivative_of_odd_is_even_l2255_225539

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem derivative_of_odd_is_even
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) (hodd : OddFunction f) :
  OddFunction f → ∀ x, deriv f (-x) = deriv f x :=
sorry

end derivative_of_odd_is_even_l2255_225539


namespace mixed_tea_sale_price_l2255_225579

/-- Represents the types of tea in the mixture -/
inductive TeaType
| First
| Second
| Third

/-- Represents the properties of each tea type -/
def tea_properties : TeaType → (Nat × Nat × Nat) :=
  fun t => match t with
  | TeaType.First  => (120, 30, 50)
  | TeaType.Second => (45, 40, 30)
  | TeaType.Third  => (35, 60, 25)

/-- Calculates the selling price for a given tea type -/
def selling_price (t : TeaType) : Nat :=
  let (weight, cost, profit) := tea_properties t
  weight * cost * (100 + profit) / 100

/-- Theorem stating the sale price of the mixed tea per kg -/
theorem mixed_tea_sale_price :
  (selling_price TeaType.First + selling_price TeaType.Second + selling_price TeaType.Third) /
  (120 + 45 + 35 : Nat) = 51825 / 1000 := by
  sorry

end mixed_tea_sale_price_l2255_225579


namespace bottle_cap_distribution_l2255_225564

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) : 
  total_caps = 35 → num_groups = 7 → total_caps = num_groups * caps_per_group → caps_per_group = 5 := by
  sorry

end bottle_cap_distribution_l2255_225564


namespace sphere_cylinder_equal_area_l2255_225511

/-- Given a sphere and a right circular cylinder with equal surface areas,
    where the cylinder has height and diameter both equal to 14 cm,
    prove that the radius of the sphere is 7 cm. -/
theorem sphere_cylinder_equal_area (r : ℝ) : 
  r > 0 → -- Ensure the radius is positive
  (4 * Real.pi * r^2 = 2 * Real.pi * 7 * 14) → -- Surface areas are equal
  r = 7 := by
  sorry

#check sphere_cylinder_equal_area

end sphere_cylinder_equal_area_l2255_225511


namespace tangent_midpoint_parallel_l2255_225594

-- Define the ellipses C and T
def ellipse_C (x y : ℝ) : Prop := x^2/18 + y^2/2 = 1
def ellipse_T (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- Define a point on an ellipse
def point_on_ellipse (E : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  E P.1 P.2

-- Define a tangent line from a point to an ellipse
def is_tangent (P M : ℝ × ℝ) (E : ℝ → ℝ → Prop) : Prop :=
  point_on_ellipse E M ∧ 
  ∀ Q, point_on_ellipse E Q → (Q ≠ M → (Q.2 - P.2) * (M.1 - P.1) ≠ (Q.1 - P.1) * (M.2 - P.2))

-- Define parallel lines
def parallel (P₁ P₂ Q₁ Q₂ : ℝ × ℝ) : Prop :=
  (P₂.2 - P₁.2) * (Q₂.1 - Q₁.1) = (P₂.1 - P₁.1) * (Q₂.2 - Q₁.2)

theorem tangent_midpoint_parallel :
  ∀ P G H M N : ℝ × ℝ,
    point_on_ellipse ellipse_C P →
    point_on_ellipse ellipse_C G →
    point_on_ellipse ellipse_C H →
    is_tangent P M ellipse_T →
    is_tangent P N ellipse_T →
    G ≠ P →
    H ≠ P →
    (G.2 - P.2) * (M.1 - P.1) = (G.1 - P.1) * (M.2 - P.2) →
    (H.2 - P.2) * (N.1 - P.1) = (H.1 - P.1) * (N.2 - P.2) →
    parallel M N G H :=
by sorry

end tangent_midpoint_parallel_l2255_225594


namespace negation_of_existence_equiv_forall_l2255_225507

theorem negation_of_existence_equiv_forall :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end negation_of_existence_equiv_forall_l2255_225507


namespace circle_radius_calculation_l2255_225581

theorem circle_radius_calculation (d PQ QR : ℝ) (h1 : d = 15) (h2 : PQ = 10) (h3 : QR = 8) :
  ∃ r : ℝ, r = 3 * Real.sqrt 5 ∧ PQ * (PQ + QR) = (d - r) * (d + r) := by
  sorry

end circle_radius_calculation_l2255_225581


namespace chocolates_bought_l2255_225563

theorem chocolates_bought (cost_price selling_price : ℝ) (num_bought : ℕ) : 
  (num_bought * cost_price = 21 * selling_price) →
  ((selling_price - cost_price) / cost_price * 100 = 66.67) →
  num_bought = 35 := by
sorry

end chocolates_bought_l2255_225563


namespace smallest_positive_integer_with_remainders_l2255_225551

theorem smallest_positive_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧ 
  b % 4 = 3 ∧ 
  b % 6 = 5 ∧ 
  (∀ c : ℕ, c > 0 ∧ c % 4 = 3 ∧ c % 6 = 5 → b ≤ c) ∧
  b = 11 :=
by sorry

end smallest_positive_integer_with_remainders_l2255_225551


namespace volume_of_specific_pyramid_l2255_225560

/-- A regular triangular pyramid with specific properties -/
structure RegularTriangularPyramid where
  /-- Distance from the midpoint of the height to the lateral face -/
  midpoint_to_face : ℝ
  /-- Distance from the midpoint of the height to the lateral edge -/
  midpoint_to_edge : ℝ

/-- The volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : ℝ := sorry

/-- Theorem stating the volume of the specific regular triangular pyramid -/
theorem volume_of_specific_pyramid :
  ∀ (p : RegularTriangularPyramid),
    p.midpoint_to_face = 2 →
    p.midpoint_to_edge = Real.sqrt 12 →
    volume p = 216 * Real.sqrt 3 := by
  sorry

end volume_of_specific_pyramid_l2255_225560


namespace subtraction_of_large_numbers_l2255_225510

theorem subtraction_of_large_numbers :
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end subtraction_of_large_numbers_l2255_225510


namespace sum_and_convert_l2255_225518

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Adds two numbers in base 8 -/
def add_base8 (a b : ℕ) : ℕ := sorry

theorem sum_and_convert :
  let a := 1453
  let b := 567
  base8_to_base10 (add_base8 a b) = 1124 := by sorry

end sum_and_convert_l2255_225518


namespace subcommittee_count_l2255_225589

theorem subcommittee_count (n m k : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 5) :
  (Nat.choose n k) - (Nat.choose (n - m) k) = 771 := by
  sorry

end subcommittee_count_l2255_225589


namespace eggs_in_box_l2255_225555

/-- The number of eggs initially in the box -/
def initial_eggs : ℝ := 47.0

/-- The number of eggs Harry adds to the box -/
def added_eggs : ℝ := 5.0

/-- The total number of eggs in the box after Harry adds eggs -/
def total_eggs : ℝ := initial_eggs + added_eggs

theorem eggs_in_box : total_eggs = 52.0 := by
  sorry

end eggs_in_box_l2255_225555


namespace halfway_between_one_seventh_and_one_ninth_l2255_225536

theorem halfway_between_one_seventh_and_one_ninth :
  (1 / 7 + 1 / 9) / 2 = 8 / 63 := by
  sorry

end halfway_between_one_seventh_and_one_ninth_l2255_225536


namespace a_gt_one_sufficient_not_necessary_l2255_225504

theorem a_gt_one_sufficient_not_necessary (a : ℝ) (h : a ≠ 0) :
  (∀ a, a > 1 → a > 1/a) ∧
  (∃ a, a > 1/a ∧ a ≤ 1) :=
sorry

end a_gt_one_sufficient_not_necessary_l2255_225504


namespace quadratic_solution_l2255_225590

theorem quadratic_solution (p q : ℝ) :
  let x : ℝ → ℝ := λ y => y - p / 2
  ∀ y, x y * x y + p * x y + q = 0 ↔ y * y = p * p / 4 - q :=
by sorry

end quadratic_solution_l2255_225590


namespace A_profit_share_l2255_225575

def investment_A : ℕ := 6300
def investment_B : ℕ := 4200
def investment_C : ℕ := 10500

def profit_share_A : ℚ := 45 / 100
def profit_share_B : ℚ := 30 / 100
def profit_share_C : ℚ := 25 / 100

def total_profit : ℕ := 12200

theorem A_profit_share :
  (profit_share_A * total_profit : ℚ) = 5490 := by sorry

end A_profit_share_l2255_225575


namespace k_range_l2255_225552

/-- The condition that for any real b, the line y = kx + b and the hyperbola x^2 - 2y^2 = 1 always have common points -/
def always_intersect (k : ℝ) : Prop :=
  ∀ b : ℝ, ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1

/-- The theorem stating the range of k given the always_intersect condition -/
theorem k_range (k : ℝ) : always_intersect k ↔ -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 := by
  sorry

end k_range_l2255_225552


namespace willie_stickers_l2255_225573

/-- Given that Willie starts with 124 stickers and gives away 43 stickers,
    prove that he ends up with 81 stickers. -/
theorem willie_stickers : ∀ (initial given_away final : ℕ),
  initial = 124 →
  given_away = 43 →
  final = initial - given_away →
  final = 81 := by
  sorry

end willie_stickers_l2255_225573


namespace auditorium_sampling_is_systematic_l2255_225534

/-- Represents a sampling method --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents an auditorium with a given number of rows and seats per row --/
structure Auditorium where
  rows : ℕ
  seatsPerRow : ℕ

/-- Represents a sampling strategy --/
structure SamplingStrategy where
  interval : ℕ
  startingSeat : ℕ

/-- Determines if a sampling strategy is systematic --/
def isSystematicSampling (strategy : SamplingStrategy) : Prop :=
  strategy.interval > 0 ∧ strategy.startingSeat > 0 ∧ strategy.startingSeat ≤ strategy.interval

/-- The theorem to be proved --/
theorem auditorium_sampling_is_systematic 
  (auditorium : Auditorium) 
  (strategy : SamplingStrategy) : 
  auditorium.rows = 25 → 
  auditorium.seatsPerRow = 20 → 
  strategy.interval = auditorium.seatsPerRow → 
  strategy.startingSeat = 15 → 
  isSystematicSampling strategy ∧ 
  SamplingMethod.Systematic = SamplingMethod.Systematic :=
sorry


end auditorium_sampling_is_systematic_l2255_225534


namespace present_age_of_A_l2255_225528

/-- Given the ages of three people A, B, and C, prove that A's present age is 11 years. -/
theorem present_age_of_A (A B C : ℕ) : 
  A + B + C = 57 → 
  ∃ (x : ℕ), A - 3 = x ∧ B - 3 = 2 * x ∧ C - 3 = 3 * x → 
  A = 11 := by
sorry

end present_age_of_A_l2255_225528


namespace solve_for_A_l2255_225566

theorem solve_for_A : ∃ A : ℕ, 3 + 68 * A = 691 ∧ 100 ≤ 68 * A ∧ 68 * A < 1000 ∧ A = 8 := by
  sorry

end solve_for_A_l2255_225566


namespace absolute_value_equation_solution_l2255_225535

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = 5 - x :=
by
  -- Proof goes here
  sorry

end absolute_value_equation_solution_l2255_225535


namespace remainder_theorem_l2255_225557

theorem remainder_theorem (n : ℤ) (h : n % 5 = 2) : (n + 2023) % 5 = 0 := by
  sorry

end remainder_theorem_l2255_225557


namespace one_intersection_point_condition_l2255_225562

open Real

noncomputable def f (x : ℝ) : ℝ := x + log x - 2 / Real.exp 1

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m / x

theorem one_intersection_point_condition (m : ℝ) :
  (∃! x, f x = g m x) →
  (m ≥ 0 ∨ m = -(Real.exp 1 + 1) / (Real.exp 1)^2) :=
by sorry

end one_intersection_point_condition_l2255_225562


namespace original_recipe_pasta_amount_l2255_225520

theorem original_recipe_pasta_amount
  (original_servings : ℕ)
  (scaled_servings : ℕ)
  (scaled_pasta : ℝ)
  (h1 : original_servings = 7)
  (h2 : scaled_servings = 35)
  (h3 : scaled_pasta = 10) :
  let pasta_per_person : ℝ := scaled_pasta / scaled_servings
  let original_pasta : ℝ := pasta_per_person * original_servings
  original_pasta = 2 := by sorry

end original_recipe_pasta_amount_l2255_225520


namespace binomial_expansion_ratio_l2255_225532

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ / a₂ = -2 := by
  sorry

end binomial_expansion_ratio_l2255_225532


namespace units_digit_of_sum_factorials_l2255_225597

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_factorials : 
  units_digit (3 * (factorial 1 + factorial 2 + factorial 3 + factorial 4)) = 9 := by
  sorry

end units_digit_of_sum_factorials_l2255_225597


namespace solve_equation_l2255_225561

theorem solve_equation (x : ℚ) (h : x / 4 - x - 3 / 6 = 1) : x = -14/9 := by
  sorry

end solve_equation_l2255_225561


namespace function_upper_bound_l2255_225524

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)

/-- A function bounded on [0,1] -/
def BoundedOnUnitInterval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x → x ≤ 1 → |f x| ≤ 1997

/-- The main theorem -/
theorem function_upper_bound
  (f : ℝ → ℝ)
  (h1 : SatisfiesInequality f)
  (h2 : BoundedOnUnitInterval f) :
  ∀ x : ℝ, x ≥ 0 → f x ≤ x^2 / 2 :=
by sorry

end function_upper_bound_l2255_225524


namespace right_triangle_hypotenuse_l2255_225549

theorem right_triangle_hypotenuse (a b : ℂ) (z₁ z₂ z₃ : ℂ) : 
  (z₁^3 + a*z₁ + b = 0) → 
  (z₂^3 + a*z₂ + b = 0) → 
  (z₃^3 + a*z₃ + b = 0) → 
  (Complex.abs z₁)^2 + (Complex.abs z₂)^2 + (Complex.abs z₃)^2 = 250 →
  ∃ (x y : ℝ), (x^2 + y^2 = (Complex.abs (z₁ - z₂))^2) ∧ 
                (x^2 = (Complex.abs (z₂ - z₃))^2 ∨ y^2 = (Complex.abs (z₂ - z₃))^2) →
  (Complex.abs (z₁ - z₂))^2 + (Complex.abs (z₂ - z₃))^2 + (Complex.abs (z₃ - z₁))^2 = 2 * ((5 * Real.sqrt 15)^2) :=
by sorry

end right_triangle_hypotenuse_l2255_225549


namespace banana_arrangement_count_l2255_225516

/-- The number of letters in the word BANANA -/
def word_length : ℕ := 6

/-- The number of occurrences of the letter B in BANANA -/
def b_count : ℕ := 1

/-- The number of occurrences of the letter N in BANANA -/
def n_count : ℕ := 2

/-- The number of occurrences of the letter A in BANANA -/
def a_count : ℕ := 3

/-- The number of unique arrangements of the letters in BANANA -/
def banana_arrangements : ℕ := word_length.factorial / (b_count.factorial * n_count.factorial * a_count.factorial)

theorem banana_arrangement_count : banana_arrangements = 60 := by
  sorry

end banana_arrangement_count_l2255_225516


namespace inequality_proof_l2255_225543

theorem inequality_proof (a b c : ℝ) (n : ℕ) (p q r : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^q * b^r * c^p + a^r * b^p * c^q :=
by sorry

end inequality_proof_l2255_225543


namespace liquid_x_percentage_in_mixed_solution_l2255_225576

/-- The percentage of liquid X in the resulting solution after mixing two solutions. -/
theorem liquid_x_percentage_in_mixed_solution
  (percent_x_in_a : ℝ)
  (percent_x_in_b : ℝ)
  (weight_a : ℝ)
  (weight_b : ℝ)
  (h1 : percent_x_in_a = 0.8)
  (h2 : percent_x_in_b = 1.8)
  (h3 : weight_a = 600)
  (h4 : weight_b = 700) :
  let weight_x_in_a := percent_x_in_a / 100 * weight_a
  let weight_x_in_b := percent_x_in_b / 100 * weight_b
  let total_weight_x := weight_x_in_a + weight_x_in_b
  let total_weight := weight_a + weight_b
  let percent_x_in_mixed := total_weight_x / total_weight * 100
  ∃ ε > 0, |percent_x_in_mixed - 1.34| < ε :=
sorry

end liquid_x_percentage_in_mixed_solution_l2255_225576


namespace trihedral_angle_range_a_trihedral_angle_range_b_l2255_225525

-- Define a trihedral angle
structure TrihedralAngle where
  α : Real
  β : Real
  γ : Real
  sum_less_than_360 : α + β + γ < 360
  each_less_than_sum_of_others : α < β + γ ∧ β < α + γ ∧ γ < α + β

-- Theorem for part (a)
theorem trihedral_angle_range_a (t : TrihedralAngle) (h1 : t.β = 70) (h2 : t.γ = 100) :
  30 < t.α ∧ t.α < 170 := by sorry

-- Theorem for part (b)
theorem trihedral_angle_range_b (t : TrihedralAngle) (h1 : t.β = 130) (h2 : t.γ = 150) :
  20 < t.α ∧ t.α < 80 := by sorry

end trihedral_angle_range_a_trihedral_angle_range_b_l2255_225525


namespace fraction_zero_implies_x_negative_one_l2255_225558

theorem fraction_zero_implies_x_negative_one (x : ℝ) : 
  (1 - |x|) / (1 - x) = 0 → x = -1 := by
  sorry

end fraction_zero_implies_x_negative_one_l2255_225558


namespace seminar_attendees_l2255_225538

theorem seminar_attendees (total : ℕ) (company_a : ℕ) : 
  total = 185 →
  company_a = 30 →
  20 = total - (company_a + 2 * company_a + (company_a + 10) + (company_a + 5)) :=
by sorry

end seminar_attendees_l2255_225538


namespace quadratic_equation_condition_l2255_225502

theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, a * x^2 = (x + 1) * (x - 1)) → a ≠ 1 := by sorry

end quadratic_equation_condition_l2255_225502


namespace helmet_safety_analysis_l2255_225533

-- Define the data types
structure YearData where
  year_number : ℕ
  not_wearing_helmets : ℕ

-- Define the data for 4 years
def year_data : List YearData := [
  ⟨1, 1250⟩,
  ⟨2, 1050⟩,
  ⟨3, 1000⟩,
  ⟨4, 900⟩
]

-- Define the contingency table
structure ContingencyTable where
  injured_not_wearing : ℕ
  injured_wearing : ℕ
  not_injured_not_wearing : ℕ
  not_injured_wearing : ℕ

def accident_data : ContingencyTable := ⟨7, 3, 13, 27⟩

-- Define the theorem
theorem helmet_safety_analysis :
  -- Regression line equation
  let b : ℚ := -110
  let a : ℚ := 1325
  let regression_line (x : ℚ) := b * x + a

  -- Estimated number of people not wearing helmets in 2022
  let estimate_2022 : ℕ := 775

  -- Chi-square statistic
  let chi_square : ℚ := 4.6875
  let critical_value : ℚ := 3.841

  -- Theorem statements
  (∀ (x : ℚ), regression_line x = b * x + a) ∧
  (regression_line 5 = estimate_2022) ∧
  (chi_square > critical_value) := by
  sorry


end helmet_safety_analysis_l2255_225533


namespace complex_difference_magnitude_l2255_225547

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 2)
  (h3 : z₁ + z₂ = Complex.mk (Real.sqrt 3) 1) :
  Complex.abs (z₁ - z₂) = 2 * Real.sqrt 3 := by sorry

end complex_difference_magnitude_l2255_225547


namespace xyz_value_l2255_225593

theorem xyz_value (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xy : x * y = 40 * Real.rpow 4 (1/3))
  (h_xz : x * z = 56 * Real.rpow 4 (1/3))
  (h_yz : y * z = 32 * Real.rpow 4 (1/3))
  (h_sum : x + y = 18) :
  x * y * z = 16 * Real.sqrt 895 := by
sorry

end xyz_value_l2255_225593


namespace intersection_implies_m_range_l2255_225514

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m/2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

def B (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2*m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2*m + 1}

-- State the theorem
theorem intersection_implies_m_range (m : ℝ) :
  (A m ∩ B m).Nonempty → 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 :=
by sorry

end intersection_implies_m_range_l2255_225514


namespace only_three_solutions_l2255_225550

/-- Represents a solution to the equation AB = B^V -/
structure Solution :=
  (a b v : Nat)
  (h1 : a ≠ b ∧ a ≠ v ∧ b ≠ v)
  (h2 : a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 ∧ v > 0 ∧ v < 10)
  (h3 : 10 * a + b = b^v)

/-- The set of all valid solutions -/
def allSolutions : Set Solution := {s | s.a > 0 ∧ s.b > 0 ∧ s.v > 0}

/-- The theorem stating that there are only three solutions -/
theorem only_three_solutions :
  allSolutions = {
    ⟨3, 2, 5, sorry, sorry, sorry⟩,
    ⟨3, 6, 2, sorry, sorry, sorry⟩,
    ⟨6, 4, 3, sorry, sorry, sorry⟩
  } := by sorry

end only_three_solutions_l2255_225550


namespace perfect_square_trinomial_k_l2255_225503

/-- If 49m^2 + km + 1 is a perfect square trinomial, then k = ±14 -/
theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ (a b : ℤ), ∀ m, 49 * m^2 + k * m + 1 = (a * m + b)^2) →
  k = 14 ∨ k = -14 := by
  sorry

end perfect_square_trinomial_k_l2255_225503


namespace natural_pythagorean_triples_real_circle_equation_l2255_225567

-- Part 1: Natural numbers
def natural_solutions : Set (ℕ × ℕ) :=
  {(0, 5), (5, 0), (3, 4), (4, 3)}

theorem natural_pythagorean_triples :
  ∀ (x y : ℕ), x^2 + y^2 = 25 ↔ (x, y) ∈ natural_solutions :=
sorry

-- Part 2: Real numbers
def real_solutions : Set (ℝ × ℝ) :=
  {(x, y) | -5 ≤ x ∧ x ≤ 5 ∧ (y = Real.sqrt (25 - x^2) ∨ y = -Real.sqrt (25 - x^2))}

theorem real_circle_equation :
  ∀ (x y : ℝ), x^2 + y^2 = 25 ↔ (x, y) ∈ real_solutions :=
sorry

end natural_pythagorean_triples_real_circle_equation_l2255_225567


namespace added_number_problem_l2255_225587

theorem added_number_problem (initial_count : ℕ) (initial_avg : ℚ) (new_avg : ℚ) : 
  initial_count = 6 →
  initial_avg = 24 →
  new_avg = 25 →
  ∃ x : ℚ, (initial_count * initial_avg + x) / (initial_count + 1) = new_avg ∧ x = 31 :=
by sorry

end added_number_problem_l2255_225587


namespace cylinder_section_area_l2255_225574

/-- The area of a plane section in a cylinder --/
theorem cylinder_section_area (r h : ℝ) (arc_angle : ℝ) : 
  r = 8 → h = 10 → arc_angle = 150 * π / 180 →
  ∃ (area : ℝ), area = (400/3) * π + 40 * Real.sqrt 3 := by
  sorry

end cylinder_section_area_l2255_225574


namespace bottle_and_beverage_weight_l2255_225578

/-- Given a bottle and some beverage, prove the weight of the original beverage and the bottle. -/
theorem bottle_and_beverage_weight 
  (original_beverage : ℝ) 
  (bottle : ℝ) 
  (h1 : 2 * original_beverage + bottle = 5) 
  (h2 : 4 * original_beverage + bottle = 9) : 
  original_beverage = 2 ∧ bottle = 1 := by
sorry

end bottle_and_beverage_weight_l2255_225578


namespace triangle_theorem_l2255_225512

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * Real.sin (t.B + t.C) + t.a * Real.cos t.B = t.c) 
  (h2 : t.a = 6)
  (h3 : t.b + t.c = 6 + 6 * Real.sqrt 3) : 
  t.A = π / 6 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = 9 * Real.sqrt 3 := by
  sorry

end triangle_theorem_l2255_225512


namespace inequality_equivalence_l2255_225513

theorem inequality_equivalence (x : ℝ) :
  (1 / x > -4 ∧ 1 / x < 3) ↔ (x > 1 / 3 ∨ x < -1 / 4) :=
by sorry

end inequality_equivalence_l2255_225513


namespace line_intersection_with_x_axis_l2255_225541

/-- Given a line y = kx + b parallel to y = -3x + 1 and passing through (0, -2),
    prove that its intersection with the x-axis is at (-2/3, 0) -/
theorem line_intersection_with_x_axis
  (k b : ℝ) 
  (parallel : k = -3)
  (passes_through : b = -2) :
  let line := λ x : ℝ => k * x + b
  ∃ x : ℝ, line x = 0 ∧ x = -2/3 :=
by sorry

end line_intersection_with_x_axis_l2255_225541


namespace nearest_integer_to_power_l2255_225595

theorem nearest_integer_to_power : 
  ∃ n : ℤ, n = 3936 ∧ ∀ m : ℤ, |((3:ℝ) + Real.sqrt 5)^5 - (n:ℝ)| ≤ |((3:ℝ) + Real.sqrt 5)^5 - (m:ℝ)| :=
by sorry

end nearest_integer_to_power_l2255_225595


namespace quadratic_equation_condition_l2255_225569

theorem quadratic_equation_condition (m : ℝ) : 
  (m ^ 2 - 7 = 2 ∧ m - 3 ≠ 0) ↔ m = -3 := by sorry

end quadratic_equation_condition_l2255_225569


namespace intersection_of_M_and_N_l2255_225554

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end intersection_of_M_and_N_l2255_225554


namespace added_amount_l2255_225546

theorem added_amount (x : ℝ) (y : ℝ) : 
  x = 15 → 3 * (2 * x + y) = 105 → y = 5 := by
  sorry

end added_amount_l2255_225546


namespace sum_of_threes_place_values_63130_l2255_225548

def number : ℕ := 63130

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def sum_of_threes_place_values (n : ℕ) : ℕ :=
  hundreds_digit n * 100 + tens_digit n * 10

theorem sum_of_threes_place_values_63130 :
  sum_of_threes_place_values number = 330 := by
  sorry

end sum_of_threes_place_values_63130_l2255_225548


namespace hedge_cost_proof_l2255_225572

/-- The number of concrete blocks used in each section of the hedge. -/
def blocks_per_section : ℕ := 30

/-- The cost of each concrete block in dollars. -/
def cost_per_block : ℕ := 2

/-- The number of sections in the hedge. -/
def number_of_sections : ℕ := 8

/-- The total cost of concrete blocks for the hedge. -/
def total_cost : ℕ := blocks_per_section * number_of_sections * cost_per_block

theorem hedge_cost_proof : total_cost = 480 := by
  sorry

end hedge_cost_proof_l2255_225572


namespace min_value_w_z_cubes_l2255_225527

/-- Given complex numbers w and z satisfying |w + z| = 1 and |w² + z²| = 14,
    the smallest possible value of |w³ + z³| is 41/2. -/
theorem min_value_w_z_cubes (w z : ℂ) 
    (h1 : Complex.abs (w + z) = 1)
    (h2 : Complex.abs (w^2 + z^2) = 14) :
    ∃ (m : ℝ), m = 41/2 ∧ ∀ (x : ℝ), x ≥ m → Complex.abs (w^3 + z^3) ≤ x :=
by sorry

end min_value_w_z_cubes_l2255_225527


namespace prob_king_ace_value_l2255_225505

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the event of drawing a King as the first card -/
def first_card_king (d : Deck) : Finset (Fin 52) :=
  d.cards.filter (λ c => c ≤ 4)

/-- Represents the event of drawing an Ace as the second card -/
def second_card_ace (d : Deck) : Finset (Fin 52) :=
  d.cards.filter (λ c => c ≤ 4 ∧ c ≠ c)

/-- The probability of drawing a King first and an Ace second -/
def prob_king_ace (d : Deck) : ℚ :=
  (first_card_king d).card * (second_card_ace d).card / (d.cards.card * (d.cards.card - 1))

theorem prob_king_ace_value (d : Deck) : prob_king_ace d = 4 / 663 := by
  sorry

end prob_king_ace_value_l2255_225505


namespace ferry_speed_difference_l2255_225592

/-- Proves the speed difference between two ferries given their travel conditions -/
theorem ferry_speed_difference :
  -- Ferry P's travel time
  let t_p : ℝ := 2 
  -- Ferry P's speed
  let v_p : ℝ := 8
  -- Ferry Q's route length multiplier
  let route_multiplier : ℝ := 3
  -- Additional time for Ferry Q's journey
  let additional_time : ℝ := 2

  -- Distance traveled by Ferry P
  let d_p : ℝ := t_p * v_p
  -- Distance traveled by Ferry Q
  let d_q : ℝ := route_multiplier * d_p
  -- Total time for Ferry Q's journey
  let t_q : ℝ := t_p + additional_time
  -- Speed of Ferry Q
  let v_q : ℝ := d_q / t_q

  -- The speed difference between Ferry Q and Ferry P is 4 km/hour
  v_q - v_p = 4 := by sorry

end ferry_speed_difference_l2255_225592


namespace gcd_lcm_product_28_45_l2255_225523

theorem gcd_lcm_product_28_45 : Nat.gcd 28 45 * Nat.lcm 28 45 = 1260 := by
  sorry

end gcd_lcm_product_28_45_l2255_225523


namespace f_inequality_l2255_225584

open Real

-- Define the function f on (0, +∞)
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State that f' is indeed the derivative of f
variable (hf' : ∀ x, x > 0 → HasDerivAt f (f' x) x)

-- State the condition xf'(x) > 2f(x)
variable (h_cond : ∀ x, x > 0 → x * f' x > 2 * f x)

-- Define a and b
variable (a b : ℝ)

-- State that a > b > 0
variable (hab : a > b ∧ b > 0)

-- Theorem statement
theorem f_inequality : b^2 * f a > a^2 * f b := by
  sorry

end f_inequality_l2255_225584


namespace five_spiders_make_five_webs_l2255_225544

/-- The number of webs made by a given number of spiders in 5 days -/
def webs_made (num_spiders : ℕ) : ℕ :=
  num_spiders * 1

/-- Theorem stating that 5 spiders make 5 webs in 5 days -/
theorem five_spiders_make_five_webs :
  webs_made 5 = 5 := by
  sorry

end five_spiders_make_five_webs_l2255_225544


namespace multiply_special_polynomials_l2255_225530

theorem multiply_special_polynomials (y : ℝ) :
  (y^4 + 30*y^2 + 900) * (y^2 - 30) = y^6 - 27000 := by
  sorry

end multiply_special_polynomials_l2255_225530


namespace envelope_equals_cycloid_l2255_225553

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle rolling along the x-axis -/
structure RollingCircle where
  radius : ℝ
  center : Point2D

/-- Represents a cycloid curve -/
def Cycloid := ℝ → Point2D

/-- Generates the cycloid traced by a point on the circumference of a circle -/
def circumferenceCycloid (radius : ℝ) : Cycloid := sorry

/-- Generates the envelope of a diameter of a rolling circle -/
def diameterEnvelope (radius : ℝ) : Cycloid := sorry

/-- Theorem stating that the envelope of a diameter is identical to the cycloid traced by a point on the circumference -/
theorem envelope_equals_cycloid (a : ℝ) :
  diameterEnvelope a = circumferenceCycloid (a / 2) := by sorry

end envelope_equals_cycloid_l2255_225553


namespace percentage_not_covering_politics_l2255_225531

/-- Represents the percentage of reporters covering local politics in country X -/
def local_politics_coverage : ℝ := 28

/-- Represents the percentage of political reporters not covering local politics in country X -/
def non_local_politics_coverage : ℝ := 30

/-- Theorem stating that 60% of reporters do not cover politics given the conditions -/
theorem percentage_not_covering_politics :
  let total_political_coverage := local_politics_coverage / (1 - non_local_politics_coverage / 100)
  100 - total_political_coverage = 60 := by
  sorry

end percentage_not_covering_politics_l2255_225531


namespace division_problem_l2255_225515

theorem division_problem (x : ℝ) : 25.25 / x = 0.012625 → x = 2000 := by
  sorry

end division_problem_l2255_225515


namespace product_from_hcf_lcm_l2255_225571

theorem product_from_hcf_lcm (A B : ℕ+) :
  Nat.gcd A B = 22 →
  Nat.lcm A B = 2828 →
  A * B = 62216 := by
  sorry

end product_from_hcf_lcm_l2255_225571


namespace quadratic_inequality_range_l2255_225540

theorem quadratic_inequality_range (m : ℝ) (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) →
  (∀ b : ℝ, b > a → m > b) →
  a = 1 := by
sorry

end quadratic_inequality_range_l2255_225540


namespace share_percentage_problem_l2255_225565

theorem share_percentage_problem (total z y x : ℝ) : 
  total = 740 →
  z = 200 →
  y = 1.2 * z →
  x = total - y - z →
  (x - y) / y * 100 = 25 := by
  sorry

end share_percentage_problem_l2255_225565


namespace circle_condition_l2255_225545

theorem circle_condition (x y m : ℝ) : 
  (∃ (a b r : ℝ), r > 0 ∧ (x - a)^2 + (y - b)^2 = r^2 ↔ x^2 + y^2 - x + y + m = 0) → 
  m < (1/2 : ℝ) :=
by sorry

end circle_condition_l2255_225545


namespace quadratic_coefficient_values_l2255_225522

/-- A quadratic function f(x) = ax^2 + 2ax + 1 with a maximum value of 5 on the interval [-2, 3] -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

/-- The maximum value of the quadratic function on the given interval -/
def max_value : ℝ := 5

/-- The lower bound of the interval -/
def lower_bound : ℝ := -2

/-- The upper bound of the interval -/
def upper_bound : ℝ := 3

/-- Theorem stating that the value of 'a' in the quadratic function
    with the given properties is either 4/15 or -4 -/
theorem quadratic_coefficient_values :
  ∃ (a : ℝ), (∀ x ∈ Set.Icc lower_bound upper_bound,
    quadratic_function a x ≤ max_value) ∧
  (∃ x ∈ Set.Icc lower_bound upper_bound,
    quadratic_function a x = max_value) ∧
  (a = 4/15 ∨ a = -4) :=
by sorry

end quadratic_coefficient_values_l2255_225522


namespace quadratic_root_equivalence_l2255_225521

theorem quadratic_root_equivalence (a b c : ℝ) (h : a ≠ 0) :
  (1 = 1 ∧ a * 1^2 + b * 1 + c = 0) ↔ (a + b + c = 0) := by
  sorry

end quadratic_root_equivalence_l2255_225521


namespace cube_inequality_l2255_225591

theorem cube_inequality (a b : ℝ) (h : a < 0 ∧ 0 < b) : a^3 < b^3 := by
  sorry

end cube_inequality_l2255_225591


namespace complex_number_modulus_l2255_225537

theorem complex_number_modulus (z : ℂ) : z = -5 + 12 * Complex.I → Complex.abs z = 13 := by
  sorry

end complex_number_modulus_l2255_225537


namespace seokjin_paper_count_prove_seokjin_paper_count_l2255_225508

theorem seokjin_paper_count : ℕ → ℕ → ℕ → Prop :=
  fun jimin_count seokjin_count difference =>
    (jimin_count = 41) →
    (seokjin_count = jimin_count - difference) →
    (difference = 1) →
    (seokjin_count = 40)

#check seokjin_paper_count

theorem prove_seokjin_paper_count :
  seokjin_paper_count 41 40 1 := by sorry

end seokjin_paper_count_prove_seokjin_paper_count_l2255_225508


namespace sum_a_c_equals_five_l2255_225588

theorem sum_a_c_equals_five 
  (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 40) 
  (h2 : b + d = 8) : 
  a + c = 5 := by sorry

end sum_a_c_equals_five_l2255_225588


namespace competition_results_l2255_225582

def team_a_scores : List ℝ := [7, 8, 9, 7, 10, 10, 9, 10, 10, 10]
def team_b_scores : List ℝ := [10, 8, 7, 9, 8, 10, 10, 9, 10, 9]

def median (scores : List ℝ) : ℝ := sorry
def mode (scores : List ℝ) : ℝ := sorry
def average (scores : List ℝ) : ℝ := sorry
def variance (scores : List ℝ) : ℝ := sorry

theorem competition_results :
  median team_a_scores = 9.5 ∧
  mode team_b_scores = 10 ∧
  average team_b_scores = 9 ∧
  variance team_b_scores = 1 ∧
  variance team_a_scores = 1.4 ∧
  variance team_b_scores < variance team_a_scores :=
by sorry

end competition_results_l2255_225582


namespace simplify_fraction_product_l2255_225577

theorem simplify_fraction_product :
  (1 : ℝ) / (1 + Real.sqrt 3) * (1 / (1 - Real.sqrt 5)) = 1 / (1 - Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 15) := by
  sorry

end simplify_fraction_product_l2255_225577


namespace boys_without_calculators_l2255_225506

/-- Given a class with boys and girls, and information about calculator possession,
    prove the number of boys without calculators. -/
theorem boys_without_calculators
  (total_boys : ℕ)
  (total_with_calculators : ℕ)
  (girls_with_calculators : ℕ)
  (h1 : total_boys = 20)
  (h2 : total_with_calculators = 26)
  (h3 : girls_with_calculators = 13) :
  total_boys - (total_with_calculators - girls_with_calculators) = 7 :=
by
  sorry

#check boys_without_calculators

end boys_without_calculators_l2255_225506


namespace min_sum_xy_l2255_225556

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y * (x - y)^2 = 1) : 
  x + y ≥ 2 := by
sorry

end min_sum_xy_l2255_225556


namespace problem_solution_l2255_225542

def p : Prop := 0 % 2 = 0
def q : Prop := ∃ k : ℤ, 3 = 2 * k

theorem problem_solution : p ∨ q := by sorry

end problem_solution_l2255_225542


namespace fractional_equation_solution_l2255_225596

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 2 ∧ (1 / (x - 2) + (1 - x) / (2 - x) = 3) ∧ x = 3 := by
  sorry

end fractional_equation_solution_l2255_225596


namespace modulus_of_z_l2255_225529

def i : ℂ := Complex.I

theorem modulus_of_z (z : ℂ) (h : z * (1 + i) = 2 - i) : Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end modulus_of_z_l2255_225529


namespace perpendicular_sequence_limit_l2255_225526

/-- An equilateral triangle ABC with a sequence of points Pₙ on AB defined by perpendicular constructions --/
structure PerpendicularSequence where
  /-- The side length of the equilateral triangle --/
  a : ℝ
  /-- The sequence of distances BPₙ --/
  bp : ℕ → ℝ
  /-- The initial point P₁ is on AB --/
  h_initial : 0 ≤ bp 1 ∧ bp 1 ≤ a
  /-- The recurrence relation for the sequence --/
  h_recurrence : ∀ n, bp (n + 1) = 3/4 * a - 1/8 * bp n

/-- The limit of the perpendicular sequence converges to 2/3 of the side length --/
theorem perpendicular_sequence_limit (ps : PerpendicularSequence) :
  ∃ L, L = 2/3 * ps.a ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, |ps.bp n - L| < ε :=
sorry

end perpendicular_sequence_limit_l2255_225526


namespace mikes_marbles_l2255_225570

/-- Given that Mike has 8 orange marbles initially and gives away 4 marbles,
    prove that he will have 4 orange marbles remaining. -/
theorem mikes_marbles (initial_marbles : ℕ) (marbles_given : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 8 →
  marbles_given = 4 →
  remaining_marbles = initial_marbles - marbles_given →
  remaining_marbles = 4 := by
sorry

end mikes_marbles_l2255_225570


namespace projectile_meeting_time_l2255_225500

theorem projectile_meeting_time : 
  let initial_distance : ℝ := 2520
  let speed1 : ℝ := 432
  let speed2 : ℝ := 576
  let combined_speed : ℝ := speed1 + speed2
  let time_hours : ℝ := initial_distance / combined_speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 150 := by sorry

end projectile_meeting_time_l2255_225500


namespace least_n_satisfying_inequality_l2255_225509

theorem least_n_satisfying_inequality : 
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (1 : ℚ) / m - (1 : ℚ) / (m + 1) < (1 : ℚ) / 15 → m ≥ n) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 :=
by sorry

end least_n_satisfying_inequality_l2255_225509


namespace trig_identity_l2255_225517

theorem trig_identity (α : ℝ) : 
  (2 * (Real.cos (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) / 
  (2 * (Real.sin (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) = 
  Real.sin (4 * α + π/6) / Real.sin (4 * α - π/6) := by
sorry

end trig_identity_l2255_225517
