import Mathlib

namespace consecutive_odd_squares_sum_l1011_101167

theorem consecutive_odd_squares_sum : ∃ x : ℤ, 
  (x - 2)^2 + x^2 + (x + 2)^2 = 5555 ∧ 
  Odd x ∧ Odd (x - 2) ∧ Odd (x + 2) := by
  sorry

end consecutive_odd_squares_sum_l1011_101167


namespace solve_exponential_equation_l1011_101120

theorem solve_exponential_equation :
  ∃ x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) ∧ x = -10 := by
  sorry

end solve_exponential_equation_l1011_101120


namespace factorization_problems_l1011_101147

theorem factorization_problems (x y : ℝ) (m : ℝ) : 
  (x^2 - 4 = (x + 2) * (x - 2)) ∧ 
  (2*m*x^2 - 4*m*x + 2*m = 2*m*(x - 1)^2) ∧ 
  ((y^2 - 1)^2 - 6*(y^2 - 1) + 9 = (y + 2)^2 * (y - 2)^2) := by
  sorry

end factorization_problems_l1011_101147


namespace certain_event_good_product_l1011_101194

/-- Represents the total number of products --/
def total_products : ℕ := 12

/-- Represents the number of good products --/
def good_products : ℕ := 10

/-- Represents the number of defective products --/
def defective_products : ℕ := 2

/-- Represents the number of products selected --/
def selected_products : ℕ := 3

/-- Represents a selection of products --/
def Selection := Fin selected_products → Fin total_products

/-- Predicate to check if a selection contains at least one good product --/
def contains_good_product (s : Selection) : Prop :=
  ∃ i, s i < good_products

/-- The main theorem stating that any selection contains at least one good product --/
theorem certain_event_good_product :
  ∀ s : Selection, contains_good_product s :=
sorry

end certain_event_good_product_l1011_101194


namespace common_ratio_of_specific_geometric_sequence_l1011_101163

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem common_ratio_of_specific_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 2/3 →
  a 4 = ∫ x in (1:ℝ)..(4:ℝ), (1 + 2*x) →
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 3 := by
sorry

end common_ratio_of_specific_geometric_sequence_l1011_101163


namespace cone_height_ratio_l1011_101193

/-- Proves the ratio of heights for a cone with reduced height and constant base --/
theorem cone_height_ratio (original_height : ℝ) (base_circumference : ℝ) (shorter_volume : ℝ) :
  original_height = 15 →
  base_circumference = 10 * Real.pi →
  shorter_volume = 50 * Real.pi →
  ∃ (shorter_height : ℝ),
    (1 / 3) * Real.pi * (base_circumference / (2 * Real.pi))^2 * shorter_height = shorter_volume ∧
    shorter_height / original_height = 2 / 5 := by
  sorry

end cone_height_ratio_l1011_101193


namespace correct_guess_probability_l1011_101182

-- Define a finite set with 4 elements
def GameOptions : Type := Fin 4

-- Define the power set of GameOptions
def PowerSet (α : Type) : Type := Set α

-- Define the number of elements in the power set of GameOptions
def NumPossibleAnswers : Nat := 2^4 - 1  -- Exclude the empty set

-- Define the probability of guessing correctly
def ProbCorrectGuess : ℚ := 1 / NumPossibleAnswers

-- Theorem statement
theorem correct_guess_probability :
  ProbCorrectGuess = 1 / 15 := by
  sorry

end correct_guess_probability_l1011_101182


namespace probability_drawing_white_ball_l1011_101176

theorem probability_drawing_white_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
  (h1 : total_balls = 15)
  (h2 : red_balls = 9)
  (h3 : white_balls = 6)
  (h4 : total_balls = red_balls + white_balls) :
  (white_balls : ℚ) / (total_balls - 1 : ℚ) = 3 / 7 := by
  sorry

end probability_drawing_white_ball_l1011_101176


namespace return_speed_calculation_l1011_101159

/-- Proves that given a round trip with specified conditions, the return speed is 30 km/hr -/
theorem return_speed_calculation (distance : ℝ) (speed_going : ℝ) (average_speed : ℝ) 
  (h1 : distance = 150)
  (h2 : speed_going = 50)
  (h3 : average_speed = 37.5) : 
  (2 * distance) / ((distance / speed_going) + (distance / ((2 * distance) / average_speed - distance / speed_going))) = 30 :=
by sorry

end return_speed_calculation_l1011_101159


namespace pond_width_pond_width_is_10_l1011_101127

/-- The width of a rectangular pond, given its length, depth, and volume of soil extracted. -/
theorem pond_width (length depth volume : ℝ) (h1 : length = 20) (h2 : depth = 5) (h3 : volume = 1000) :
  volume = length * depth * (volume / (length * depth)) :=
by sorry

/-- The width of the pond is 10 meters. -/
theorem pond_width_is_10 (length depth volume : ℝ) (h1 : length = 20) (h2 : depth = 5) (h3 : volume = 1000) :
  volume / (length * depth) = 10 :=
by sorry

end pond_width_pond_width_is_10_l1011_101127


namespace regular_septagon_interior_angle_measure_l1011_101140

/-- The number of sides in a septagon -/
def n : ℕ := 7

/-- A regular septagon is a polygon with 7 sides and all interior angles equal -/
structure RegularSeptagon where
  sides : Fin n → ℝ
  angles : Fin n → ℝ
  all_sides_equal : ∀ i j : Fin n, sides i = sides j
  all_angles_equal : ∀ i j : Fin n, angles i = angles j

/-- Theorem: The measure of each interior angle in a regular septagon is 900/7 degrees -/
theorem regular_septagon_interior_angle_measure (s : RegularSeptagon) :
  ∀ i : Fin n, s.angles i = 900 / 7 := by
  sorry

end regular_septagon_interior_angle_measure_l1011_101140


namespace intersection_slope_l1011_101137

/-- Given two circles in the xy-plane, this theorem states that the slope of the line
    passing through their intersection points is 1/7. -/
theorem intersection_slope (x y : ℝ) :
  (x^2 + y^2 - 6*x + 4*y - 20 = 0) →
  (x^2 + y^2 - 8*x + 18*y + 40 = 0) →
  (∃ (m : ℝ), m = 1/7 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 20 = 0) →
    (x₁^2 + y₁^2 - 8*x₁ + 18*y₁ + 40 = 0) →
    (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 20 = 0) →
    (x₂^2 + y₂^2 - 8*x₂ + 18*y₂ + 40 = 0) →
    x₁ ≠ x₂ →
    m = (y₂ - y₁) / (x₂ - x₁)) :=
by sorry


end intersection_slope_l1011_101137


namespace polygon_sides_l1011_101152

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1080 → n = 8 := by
  sorry

end polygon_sides_l1011_101152


namespace power_function_through_point_l1011_101126

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x ^ a) →
  f 27 = 3 →
  a = 1 / 3 := by
sorry

end power_function_through_point_l1011_101126


namespace arithmetic_sequence_first_term_l1011_101185

/-- The sum of the first n terms of an arithmetic sequence -/
def T (a d : ℚ) (n : ℕ+) : ℚ := n * (2 * a + (n - 1) * d) / 2

/-- The theorem states that if T_{4n} / T_n is constant for an arithmetic sequence
    with common difference 5, then the first term of the sequence is 5/2 -/
theorem arithmetic_sequence_first_term
  (h : ∃ (k : ℚ), ∀ (n : ℕ+), T a 5 (4 * n) / T a 5 n = k) :
  a = 5 / 2 := by
  sorry


end arithmetic_sequence_first_term_l1011_101185


namespace hex_to_decimal_l1011_101123

/-- Given a hexadecimal number 10k5₍₆₎ where k is a positive integer,
    if this number equals 239 when converted to decimal, then k = 3. -/
theorem hex_to_decimal (k : ℕ+) : (1 * 6^3 + k * 6 + 5 = 239) → k = 3 := by
  sorry

end hex_to_decimal_l1011_101123


namespace work_increase_per_person_l1011_101180

/-- Calculates the increase in work per person when 1/6 of the workforce is absent -/
theorem work_increase_per_person (p : ℕ) (W : ℝ) (h : p > 0) :
  let initial_work_per_person := W / p
  let remaining_workers := p - p / 6
  let new_work_per_person := W / remaining_workers
  new_work_per_person - initial_work_per_person = W / (5 * p) :=
by sorry

end work_increase_per_person_l1011_101180


namespace cosine_sum_equals_radius_ratio_l1011_101145

-- Define a triangle with its angles, circumradius, and inradius
structure Triangle where
  α : Real
  β : Real
  γ : Real
  R : Real
  r : Real
  angle_sum : α + β + γ = Real.pi
  positive_R : R > 0
  positive_r : r > 0

-- State the theorem
theorem cosine_sum_equals_radius_ratio (t : Triangle) :
  Real.cos t.α + Real.cos t.β + Real.cos t.γ = (t.R + t.r) / t.R :=
by sorry

end cosine_sum_equals_radius_ratio_l1011_101145


namespace slower_speed_calculation_l1011_101156

theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) (slower_speed : ℝ) :
  actual_distance = 24 →
  faster_speed = 5 →
  additional_distance = 6 →
  faster_speed * (actual_distance / slower_speed) = actual_distance + additional_distance →
  slower_speed = 4 := by
sorry

end slower_speed_calculation_l1011_101156


namespace tangent_line_at_x_1_unique_a_for_nonnegative_f_l1011_101184

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) - (a * x) / (1 + x)

theorem tangent_line_at_x_1 (h : ℝ) :
  ∃ (m b : ℝ), ∀ x, (f 2 x - (f 2 1)) = m * (x - 1) + b ∧ 
  m * x + b = Real.log 2 - 1 := by sorry

theorem unique_a_for_nonnegative_f :
  ∃! a : ℝ, ∀ x : ℝ, x > -1 → f a x ≥ 0 := by sorry

end tangent_line_at_x_1_unique_a_for_nonnegative_f_l1011_101184


namespace binomial_variance_example_l1011_101132

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The variance of X ~ B(8, 0.7) is 1.68 -/
theorem binomial_variance_example :
  let X : BinomialRV := ⟨8, 0.7, by norm_num⟩
  variance X = 1.68 := by
  sorry

end binomial_variance_example_l1011_101132


namespace school_vote_problem_l1011_101170

theorem school_vote_problem (U A B : Finset Nat) : 
  Finset.card U = 250 →
  Finset.card A = 175 →
  Finset.card B = 140 →
  Finset.card (U \ (A ∪ B)) = 45 →
  Finset.card (A ∩ B) = 110 := by
sorry

end school_vote_problem_l1011_101170


namespace range_of_m_l1011_101197

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3) ∧ 
  (∃ x : ℝ, x > 2 * m^2 - 3 ∧ (x ≤ -1 ∨ x ≥ 4)) → 
  -1 ≤ m ∧ m ≤ 1 :=
by sorry

end range_of_m_l1011_101197


namespace common_solution_of_linear_system_l1011_101131

theorem common_solution_of_linear_system :
  (∀ (a b : ℚ), ∃ (x y : ℚ), (a - b) * x - (a + b) * y = a + b) →
  (∃! (x y : ℚ), ∀ (a b : ℚ), (a - b) * x - (a + b) * y = a + b ∧ x = 0 ∧ y = -1) :=
by sorry

end common_solution_of_linear_system_l1011_101131


namespace equilateral_triangle_reflection_parity_l1011_101199

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents a reflection of a triangle -/
def reflect (t : Triangle) : Triangle := sorry

/-- Predicate to check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Predicate to check if two triangles coincide -/
def coincide (t1 t2 : Triangle) : Prop := sorry

/-- Theorem: If an equilateral triangle is reflected multiple times and 
    coincides with the original, the number of reflections is even -/
theorem equilateral_triangle_reflection_parity 
  (t : Triangle) (n : ℕ) (h1 : is_equilateral t) :
  (coincide ((reflect^[n]) t) t) → Even n := by
  sorry

end equilateral_triangle_reflection_parity_l1011_101199


namespace mary_flour_problem_l1011_101154

theorem mary_flour_problem (recipe_flour : ℕ) (flour_to_add : ℕ) 
  (h1 : recipe_flour = 7)
  (h2 : flour_to_add = 5) :
  recipe_flour - flour_to_add = 2 := by
  sorry

end mary_flour_problem_l1011_101154


namespace arithmetic_evaluation_l1011_101196

theorem arithmetic_evaluation : 
  -(18 / 3 * 11 - 48 / 4 + 5 * 9) = -99 := by
  sorry

end arithmetic_evaluation_l1011_101196


namespace largest_after_erasing_100_l1011_101104

/-- Concatenates numbers from 1 to n as a string -/
def concatenateNumbers (n : ℕ) : String :=
  (List.range n).map (fun i => toString (i + 1)) |> String.join

/-- Checks if a number is the largest possible after erasing digits -/
def isLargestAfterErasing (original : String) (erased : ℕ) (result : String) : Prop :=
  result.length = original.length - erased ∧
  ∀ (other : String), other.length = original.length - erased →
    other.toNat! ≤ result.toNat!

theorem largest_after_erasing_100 :
  isLargestAfterErasing (concatenateNumbers 60) 100 "99999785960" := by
  sorry

end largest_after_erasing_100_l1011_101104


namespace min_value_cos_sin_l1011_101142

theorem min_value_cos_sin (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 3 * Real.pi / 2) :
  ∃ m : Real, m = -1/2 ∧ ∀ θ' : Real, 0 ≤ θ' ∧ θ' ≤ 3 * Real.pi / 2 →
    m ≤ Real.cos (θ' / 3) * (1 - Real.sin θ') :=
by sorry

end min_value_cos_sin_l1011_101142


namespace range_of_a_l1011_101125

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + (a + 2) * x + 1 ≥ 0) → -4 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l1011_101125


namespace sally_seashell_earnings_l1011_101139

-- Define the number of seashells picked on Monday
def monday_seashells : ℕ := 30

-- Define the number of seashells picked on Tuesday
def tuesday_seashells : ℕ := monday_seashells / 2

-- Define the price of each seashell in cents
def seashell_price : ℕ := 120

-- Theorem statement
theorem sally_seashell_earnings :
  (monday_seashells + tuesday_seashells) * seashell_price = 5400 := by
  sorry

end sally_seashell_earnings_l1011_101139


namespace business_valuation_l1011_101177

def business_value (total_ownership : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) : ℕ :=
  (2 * sale_price : ℕ)

theorem business_valuation (total_ownership : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) 
  (h1 : total_ownership = 2/3)
  (h2 : sold_fraction = 3/4)
  (h3 : sale_price = 6500) :
  business_value total_ownership sold_fraction sale_price = 13000 := by
  sorry

end business_valuation_l1011_101177


namespace unique_single_solution_quadratic_l1011_101112

theorem unique_single_solution_quadratic :
  ∃! (p : ℝ), p ≠ 0 ∧ (∃! x : ℝ, p * x^2 - 12 * x + 4 = 0) :=
by
  -- The proof goes here
  sorry

end unique_single_solution_quadratic_l1011_101112


namespace secret_spread_reaches_target_target_day_minimal_l1011_101175

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when the secret reaches 3280 students -/
def target_day : ℕ := 7

theorem secret_spread_reaches_target :
  secret_spread target_day = 3280 :=
sorry

theorem target_day_minimal :
  ∀ k < target_day, secret_spread k < 3280 :=
sorry

end secret_spread_reaches_target_target_day_minimal_l1011_101175


namespace real_estate_investment_l1011_101179

def total_investment : ℝ := 200000
def real_estate_ratio : ℝ := 7

theorem real_estate_investment (mutual_funds : ℝ) 
  (h1 : mutual_funds + real_estate_ratio * mutual_funds = total_investment) :
  real_estate_ratio * mutual_funds = 175000 := by
  sorry

end real_estate_investment_l1011_101179


namespace N_subset_M_l1011_101195

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 ≥ x}
def N : Set ℝ := {x : ℝ | Real.log (x + 1) / Real.log (1/2) > 0}

-- State the theorem
theorem N_subset_M : N ⊆ M := by
  sorry

end N_subset_M_l1011_101195


namespace vector_difference_magnitude_l1011_101151

def vector_a : Fin 2 → ℝ := ![2, -1]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![6, x]

theorem vector_difference_magnitude 
  (h_parallel : ∃ (k : ℝ), ∀ i, vector_a i = k * vector_b x i) :
  ∃ (x : ℝ), ‖vector_a - vector_b x‖ = 2 * Real.sqrt 5 := by
  sorry

end vector_difference_magnitude_l1011_101151


namespace interval_length_implies_k_l1011_101110

theorem interval_length_implies_k (k : ℝ) : 
  k > 0 → 
  (Set.Icc (-3) 3 : Set ℝ) = {x : ℝ | x^2 + k * |x| ≤ 2019} → 
  k = 670 := by
sorry

end interval_length_implies_k_l1011_101110


namespace concentric_circles_chords_l1011_101115

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the number of chords needed to complete
    a full circle is 3. -/
theorem concentric_circles_chords (angle : ℝ) (n : ℕ) : 
  angle = 60 → n * angle = 360 → n = 3 := by sorry

end concentric_circles_chords_l1011_101115


namespace max_pages_copied_l1011_101192

-- Define the cost per 2 pages in cents
def cost_per_2_pages : ℕ := 7

-- Define the fixed fee in cents
def fixed_fee : ℕ := 500

-- Define the total budget in cents
def total_budget : ℕ := 3500

-- Define the function to calculate the number of pages
def pages_copied (budget : ℕ) : ℕ :=
  ((budget - fixed_fee) * 2) / cost_per_2_pages

-- Theorem statement
theorem max_pages_copied :
  pages_copied total_budget = 857 := by
  sorry

end max_pages_copied_l1011_101192


namespace cylinder_max_volume_ratio_l1011_101102

/-- Given a rectangle with perimeter 12 that forms a cylinder, prove that the ratio of the base circumference to height is 2:1 when volume is maximized -/
theorem cylinder_max_volume_ratio (l w : ℝ) : 
  l > 0 → w > 0 → 
  2 * l + 2 * w = 12 → 
  let r := l / (2 * Real.pi)
  let h := w
  let V := Real.pi * r^2 * h
  (∀ l' w', l' > 0 → w' > 0 → 2 * l' + 2 * w' = 12 → 
    let r' := l' / (2 * Real.pi)
    let h' := w'
    Real.pi * r'^2 * h' ≤ V) →
  l / w = 2 := by
sorry

end cylinder_max_volume_ratio_l1011_101102


namespace molecular_weight_calculation_l1011_101113

theorem molecular_weight_calculation (total_weight : ℝ) (number_of_moles : ℝ) 
  (h1 : total_weight = 2376)
  (h2 : number_of_moles = 8) : 
  total_weight / number_of_moles = 297 := by
sorry

end molecular_weight_calculation_l1011_101113


namespace clothes_expenditure_fraction_l1011_101136

theorem clothes_expenditure_fraction (initial_amount : ℝ) (remaining_amount : ℝ) (F : ℝ) : 
  initial_amount = 499.9999999999999 →
  remaining_amount = 200 →
  remaining_amount = (3/5) * (1 - F) * initial_amount →
  F = 1/3 := by
  sorry

end clothes_expenditure_fraction_l1011_101136


namespace simplify_polynomial_l1011_101198

theorem simplify_polynomial (r : ℝ) : (2*r^2 + 5*r - 7) - (r^2 + 4*r - 6) = r^2 + r - 1 := by
  sorry

end simplify_polynomial_l1011_101198


namespace focus_of_symmetric_parabola_l1011_101183

/-- The focus of a parabola symmetric to x^2 = 4y with respect to x + y = 0 -/
def symmetric_parabola_focus : ℝ × ℝ :=
  (-1, 0)

/-- The original parabola equation -/
def original_parabola (x y : ℝ) : Prop :=
  x^2 = 4*y

/-- The line of symmetry equation -/
def symmetry_line (x y : ℝ) : Prop :=
  x + y = 0

theorem focus_of_symmetric_parabola :
  symmetric_parabola_focus = (-1, 0) :=
sorry

end focus_of_symmetric_parabola_l1011_101183


namespace range_of_m_l1011_101174

-- Define the conditions
def p (x : ℝ) : Prop := abs x > 1
def q (x m : ℝ) : Prop := x < m

-- Define the relationship between ¬p and ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  ∀ x, ¬(q x m) → ¬(p x) ∧ ∃ y, ¬(p y) ∧ q y m

-- Theorem statement
theorem range_of_m (m : ℝ) :
  not_p_necessary_not_sufficient_for_not_q m →
  m ∈ Set.Iic (-1 : ℝ) :=
sorry

end range_of_m_l1011_101174


namespace tan_half_angle_l1011_101148

theorem tan_half_angle (α : Real) 
  (h1 : π < α ∧ α < 3*π/2) 
  (h2 : Real.sin (3*π/2 + α) = 4/5) : 
  Real.tan (α/2) = -1/3 := by
sorry

end tan_half_angle_l1011_101148


namespace multiply_powers_l1011_101109

theorem multiply_powers (a : ℝ) : 2 * a^3 * 3 * a^2 = 6 * a^5 := by
  sorry

end multiply_powers_l1011_101109


namespace distribution_count_l1011_101160

-- Define the number of balls and boxes
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Define the Stirling number of the second kind function
noncomputable def stirling_second (n k : ℕ) : ℕ :=
  sorry  -- Implementation of Stirling number of the second kind

-- Theorem statement
theorem distribution_count :
  (stirling_second num_balls num_boxes) = 25 :=
sorry

end distribution_count_l1011_101160


namespace symmetry_sum_l1011_101106

/-- Two points are symmetric about the y-axis if their x-coordinates are negatives of each other
    and their y-coordinates are equal. -/
def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_about_y_axis (a, 5) (2, b) → a + b = 3 := by
  sorry

end symmetry_sum_l1011_101106


namespace joan_apples_l1011_101146

/-- The number of apples Joan gave to Melanie -/
def apples_given : ℕ := 27

/-- The number of apples Joan has now -/
def apples_left : ℕ := 16

/-- The total number of apples Joan picked from the orchard -/
def total_apples : ℕ := apples_given + apples_left

theorem joan_apples : total_apples = 43 := by
  sorry

end joan_apples_l1011_101146


namespace cookies_eaten_l1011_101135

/-- Given a package of cookies where some were eaten, this theorem proves
    the number of cookies eaten, given the initial count and remaining count. -/
theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (h : initial = 18) (h' : remaining = 9) :
  initial - remaining = 9 := by
  sorry

end cookies_eaten_l1011_101135


namespace binomial_20_10_l1011_101118

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 43758) 
                       (h2 : Nat.choose 18 9 = 48620) 
                       (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 184756 := by
  sorry

end binomial_20_10_l1011_101118


namespace married_men_fraction_l1011_101164

theorem married_men_fraction (total_women : ℕ) (total_people : ℕ) 
  (h1 : total_women > 0)
  (h2 : total_people > total_women)
  (h3 : (3 : ℚ) / 7 = (total_women - (total_people - total_women)) / total_women) :
  (total_people - total_women : ℚ) / total_people = 4 / 11 := by
sorry

end married_men_fraction_l1011_101164


namespace rahul_deepak_age_ratio_l1011_101150

theorem rahul_deepak_age_ratio : 
  ∀ (rahul_age deepak_age : ℕ),
  deepak_age = 3 →
  rahul_age + 22 = 26 →
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 := by
sorry

end rahul_deepak_age_ratio_l1011_101150


namespace intersection_of_A_and_B_l1011_101155

-- Define the sets A and B
def A : Set ℝ := {x | (2*x + 3)/(x - 2) > 0}
def B : Set ℝ := {x | |x - 1| < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l1011_101155


namespace percentage_calculation_l1011_101144

theorem percentage_calculation (x y : ℝ) : 
  x = 0.8 * 350 → y = 0.6 * x → 1.2 * y = 201.6 := by
  sorry

end percentage_calculation_l1011_101144


namespace intersection_when_a_is_one_subset_condition_l1011_101105

-- Define set A as the solution set of -x^2 - 2x + 8 = 0
def A : Set ℝ := {x | -x^2 - 2*x + 8 = 0}

-- Define set B as the solution set of ax - 1 ≤ 0
def B (a : ℝ) : Set ℝ := {x | a*x - 1 ≤ 0}

-- Theorem 1: When a = 1, A ∩ B = {-4}
theorem intersection_when_a_is_one : A ∩ B 1 = {-4} := by sorry

-- Theorem 2: A ⊆ B if and only if -1/4 ≤ a ≤ 1/2
theorem subset_condition : 
  ∀ a : ℝ, A ⊆ B a ↔ -1/4 ≤ a ∧ a ≤ 1/2 := by sorry

end intersection_when_a_is_one_subset_condition_l1011_101105


namespace factor_expression_l1011_101138

theorem factor_expression : ∀ x : ℝ, 75 * x + 45 = 15 * (5 * x + 3) := by
  sorry

end factor_expression_l1011_101138


namespace lower_right_is_three_l1011_101169

/-- Represents a 5x5 grid with digits from 1 to 5 -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Checks if a number is unique in its row -/
def unique_in_row (g : Grid) (row col : Fin 5) : Prop :=
  ∀ c : Fin 5, c ≠ col → g row c ≠ g row col

/-- Checks if a number is unique in its column -/
def unique_in_col (g : Grid) (row col : Fin 5) : Prop :=
  ∀ r : Fin 5, r ≠ row → g r col ≠ g row col

/-- Checks if the grid satisfies the uniqueness conditions -/
def valid_grid (g : Grid) : Prop :=
  ∀ r c : Fin 5, unique_in_row g r c ∧ unique_in_col g r c

/-- The theorem to be proved -/
theorem lower_right_is_three (g : Grid) 
  (h1 : valid_grid g)
  (h2 : g 0 0 = 1)
  (h3 : g 0 4 = 2)
  (h4 : g 1 1 = 4)
  (h5 : g 2 3 = 3)
  (h6 : g 3 2 = 5) :
  g 4 4 = 3 := by
  sorry

end lower_right_is_three_l1011_101169


namespace hanging_spheres_mass_ratio_l1011_101119

/-- Given two hanging spheres with masses m₁ and m₂, where the tension in the upper string
    is twice the tension in the lower string, prove that the ratio of masses m₁/m₂ = 1 -/
theorem hanging_spheres_mass_ratio (m₁ m₂ : ℝ) (g : ℝ) (h : g > 0) : 
  (m₁ * g + m₂ * g = 2 * (m₂ * g)) → m₁ / m₂ = 1 := by
  sorry

end hanging_spheres_mass_ratio_l1011_101119


namespace arg_z_range_l1011_101189

theorem arg_z_range (z : ℂ) (h : |Complex.arg ((z + 1) / (z + 2))| = π / 6) :
  Complex.arg z ∈ Set.union
    (Set.Ioo (5 * π / 6 - Real.arcsin (Real.sqrt 3 / 3)) π)
    (Set.Ioo π (7 * π / 6 + Real.arcsin (Real.sqrt 3 / 3))) := by
  sorry

end arg_z_range_l1011_101189


namespace mike_bought_21_books_l1011_101107

/-- The number of books Mike bought at a yard sale -/
def books_bought (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem stating that Mike bought 21 books at the yard sale -/
theorem mike_bought_21_books :
  books_bought 35 56 = 21 := by
  sorry

end mike_bought_21_books_l1011_101107


namespace common_roots_product_l1011_101141

theorem common_roots_product (C : ℝ) : 
  ∃ (p q r t : ℝ), 
    (p^3 + 2*p^2 + 15 = 0) ∧ 
    (q^3 + 2*q^2 + 15 = 0) ∧ 
    (r^3 + 2*r^2 + 15 = 0) ∧
    (p^3 + C*p + 30 = 0) ∧ 
    (q^3 + C*q + 30 = 0) ∧ 
    (t^3 + C*t + 30 = 0) ∧
    (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ 
    (p ≠ t) ∧ (q ≠ t) →
    p * q = -5 * Real.rpow 2 (1/3) :=
by sorry

end common_roots_product_l1011_101141


namespace compute_expression_l1011_101143

theorem compute_expression : 8 * (1 / 4)^4 = 1 / 32 := by
  sorry

end compute_expression_l1011_101143


namespace triangle_existence_condition_l1011_101157

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_x_values : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

theorem triangle_existence_condition (x : ℕ) :
  x > 0 → (triangle_exists 7 (x + 3) 10 ↔ x ∈ valid_x_values) := by sorry

end triangle_existence_condition_l1011_101157


namespace distance_solution_l1011_101187

/-- The distance from a dormitory to a city -/
def distance_problem (D : ℝ) : Prop :=
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 12 = D

theorem distance_solution : ∃ D : ℝ, distance_problem D ∧ D = 90 := by
  sorry

end distance_solution_l1011_101187


namespace square_area_triple_l1011_101108

/-- Given a square I with diagonal 2a, prove that a square II with triple the area of square I has an area of 6a² -/
theorem square_area_triple (a : ℝ) :
  let diagonal_I : ℝ := 2 * a
  let area_I : ℝ := (diagonal_I ^ 2) / 2
  let area_II : ℝ := 3 * area_I
  area_II = 6 * a ^ 2 := by
sorry

end square_area_triple_l1011_101108


namespace six_balls_three_boxes_l1011_101181

/-- The number of ways to partition n indistinguishable objects into k or fewer non-empty, indistinguishable groups -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to partition 6 indistinguishable objects into 3 or fewer non-empty, indistinguishable groups -/
theorem six_balls_three_boxes : partition_count 6 3 = 6 := by sorry

end six_balls_three_boxes_l1011_101181


namespace x_value_proof_l1011_101178

theorem x_value_proof : ∃ x : ℝ, x = 70 * (1 + 11/100) ∧ x = 77.7 := by
  sorry

end x_value_proof_l1011_101178


namespace factorial_fraction_equals_seven_l1011_101133

theorem factorial_fraction_equals_seven : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 7 := by
  sorry

end factorial_fraction_equals_seven_l1011_101133


namespace inequality_proof_l1011_101165

theorem inequality_proof (x y : ℝ) : (1 / 2) * (x^2 + y^2) - x * y ≥ 0 := by
  sorry

end inequality_proof_l1011_101165


namespace parallel_angles_theorem_l1011_101171

theorem parallel_angles_theorem (α β : Real) :
  (∃ k : ℤ, α + β = k * 180) →  -- Parallel sides condition
  (α = 3 * β - 36) →            -- Relationship between α and β
  (α = 18 ∨ α = 126) :=         -- Conclusion
by sorry

end parallel_angles_theorem_l1011_101171


namespace hyperbola_conjugate_axis_length_l1011_101129

theorem hyperbola_conjugate_axis_length :
  ∀ (x y : ℝ), 2 * x^2 - y^2 = 8 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (x^2 / a^2) - (y^2 / b^2) = 1 ∧
  2 * b = 4 * Real.sqrt 2 :=
by sorry

end hyperbola_conjugate_axis_length_l1011_101129


namespace tangent_is_simson_line_l1011_101101

/-- A parabola in a 2D plane. -/
structure Parabola where
  -- Add necessary fields to define a parabola

/-- A triangle in a 2D plane. -/
structure Triangle where
  -- Add necessary fields to define a triangle

/-- The Simson line of a triangle with respect to a point. -/
def SimsonLine (t : Triangle) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- The tangent line to a parabola at a given point. -/
def TangentLine (p : Parabola) (point : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- The vertex of a parabola. -/
def Vertex (p : Parabola) : ℝ × ℝ :=
  sorry

/-- Given three tangent lines to a parabola, find their intersection points forming a triangle. -/
def TriangleFromTangents (p : Parabola) (t1 t2 t3 : Set (ℝ × ℝ)) : Triangle :=
  sorry

theorem tangent_is_simson_line (p : Parabola) (t1 t2 t3 : Set (ℝ × ℝ)) :
  TangentLine p (Vertex p) = SimsonLine (TriangleFromTangents p t1 t2 t3) (Vertex p) :=
sorry

end tangent_is_simson_line_l1011_101101


namespace total_votes_correct_l1011_101162

/-- The total number of votes in an election --/
def total_votes : ℕ := 560000

/-- The percentage of valid votes that Candidate A received --/
def candidate_A_percentage : ℚ := 55 / 100

/-- The percentage of invalid votes --/
def invalid_percentage : ℚ := 15 / 100

/-- The number of valid votes Candidate A received --/
def candidate_A_votes : ℕ := 261800

/-- Theorem stating that the total number of votes is correct given the conditions --/
theorem total_votes_correct :
  (↑candidate_A_votes : ℚ) = 
    (1 - invalid_percentage) * candidate_A_percentage * (↑total_votes : ℚ) :=
sorry

end total_votes_correct_l1011_101162


namespace smallest_n_square_cube_l1011_101111

theorem smallest_n_square_cube : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 ∧ x < n → ¬(∃ (y : ℕ), 4 * x = y^2) ∨ ¬(∃ (z : ℕ), 5 * x = z^3)) ∧
  n = 100 :=
by sorry

end smallest_n_square_cube_l1011_101111


namespace arctan_sum_three_four_l1011_101149

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end arctan_sum_three_four_l1011_101149


namespace tan_sum_product_l1011_101188

theorem tan_sum_product (α β : Real) (h : α + β = 3 * Real.pi / 4) :
  (1 - Real.tan α) * (1 - Real.tan β) = 2 := by
  sorry

end tan_sum_product_l1011_101188


namespace calculation_proof_l1011_101166

theorem calculation_proof : Real.sqrt 4 - Real.sin (30 * π / 180) - (π - 1) ^ 0 + 2⁻¹ = 1 := by
  sorry

end calculation_proof_l1011_101166


namespace price_reduction_theorem_l1011_101116

/-- Proves that a price reduction resulting in an 80% increase in sales and a 26% increase in total revenue implies a 30% price reduction -/
theorem price_reduction_theorem (P S : ℝ) (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 100) 
  (h3 : P > 0) 
  (h4 : S > 0) 
  (h5 : P * (1 - x / 100) * (S * 1.8) = P * S * 1.26) : 
  x = 30 := by
  sorry

end price_reduction_theorem_l1011_101116


namespace museum_artifacts_l1011_101134

theorem museum_artifacts (total_wings : ℕ) (painting_wings : ℕ) (large_painting : ℕ) 
  (small_paintings_per_wing : ℕ) (artifact_multiplier : ℕ) :
  total_wings = 8 →
  painting_wings = 3 →
  large_painting = 1 →
  small_paintings_per_wing = 12 →
  artifact_multiplier = 4 →
  let total_paintings := large_painting + 2 * small_paintings_per_wing
  let total_artifacts := artifact_multiplier * total_paintings
  let artifact_wings := total_wings - painting_wings
  total_artifacts / artifact_wings = 20 :=
by sorry

end museum_artifacts_l1011_101134


namespace license_plate_increase_l1011_101124

theorem license_plate_increase : 
  let old_plates := 26 * 10^3
  let new_plates := 26^2 * 10^4
  new_plates / old_plates = 260 := by
sorry

end license_plate_increase_l1011_101124


namespace A_eq_real_iff_m_in_range_l1011_101161

/-- The set A defined by the quadratic inequality -/
def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 + 2 * m * x + 1 > 0}

/-- Theorem stating the equivalence between A being equal to ℝ and m being in [0, 1) -/
theorem A_eq_real_iff_m_in_range (m : ℝ) : A m = Set.univ ↔ m ∈ Set.Icc 0 1 := by
  sorry

end A_eq_real_iff_m_in_range_l1011_101161


namespace tan_simplification_l1011_101172

theorem tan_simplification (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + 3 * Real.sin α) = 3/8 := by
  sorry

end tan_simplification_l1011_101172


namespace problem_solution_l1011_101168

/-- Given a function f(x) = x² - 2x + 2a, where the solution set of f(x) ≤ 0 is {x | -2 ≤ x ≤ m},
    prove that a = -4 and m = 4, and find the range of c where (c+a)x² + 2(c+a)x - 1 < 0 always holds for x. -/
theorem problem_solution (a m : ℝ) (f : ℝ → ℝ) (c : ℝ) : 
  (f = fun x => x^2 - 2*x + 2*a) →
  (∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ m) →
  (a = -4 ∧ m = 4) ∧
  (∀ x, (c + a)*x^2 + 2*(c + a)*x - 1 < 0 ↔ 13/4 < c ∧ c < 4) :=
by sorry

end problem_solution_l1011_101168


namespace orthocenter_from_circumcenter_l1011_101114

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Checks if a point is the circumcenter of a triangle -/
def isCircumcenter (O : Point3D) (A B C : Point3D) : Prop := sorry

/-- Checks if a point is the orthocenter of a triangle -/
def isOrthocenter (H : Point3D) (A B C : Point3D) : Prop := sorry

/-- Checks if a sphere is inscribed in a tetrahedron -/
def isInscribed (s : Sphere) (t : Tetrahedron) : Prop := sorry

/-- Checks if a sphere touches a plane at a point -/
def touchesPlaneAt (s : Sphere) (p : Point3D) : Prop := sorry

/-- Checks if a sphere touches the planes of the other faces of a tetrahedron externally -/
def touchesOtherFacesExternally (s : Sphere) (t : Tetrahedron) : Prop := sorry

theorem orthocenter_from_circumcenter 
  (t : Tetrahedron) 
  (s1 s2 : Sphere) 
  (H O : Point3D) :
  isInscribed s1 t →
  touchesPlaneAt s1 H →
  touchesPlaneAt s2 O →
  touchesOtherFacesExternally s2 t →
  isCircumcenter O t.A t.B t.C →
  isOrthocenter H t.A t.B t.C := by
  sorry

end orthocenter_from_circumcenter_l1011_101114


namespace unique_permutations_four_letter_two_pairs_is_six_l1011_101158

/-- The number of unique permutations of a four-letter word with two pairs of identical letters -/
def unique_permutations_four_letter_two_pairs : ℕ :=
  Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem: The number of unique permutations of a four-letter word with two pairs of identical letters is 6 -/
theorem unique_permutations_four_letter_two_pairs_is_six :
  unique_permutations_four_letter_two_pairs = 6 := by
  sorry

end unique_permutations_four_letter_two_pairs_is_six_l1011_101158


namespace water_to_height_ratio_l1011_101191

def rons_height : ℝ := 12
def water_depth : ℝ := 60

theorem water_to_height_ratio : water_depth / rons_height = 5 := by
  sorry

end water_to_height_ratio_l1011_101191


namespace parallel_line_through_point_l1011_101186

/-- A line passing through (1,0) parallel to x-2y-2=0 has equation x-2y-1=0 -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y - 1 = 0) ↔ 
  (∃ (m b : ℝ), y = m*x + b ∧ 
                 m = (1 : ℝ)/(2 : ℝ) ∧ 
                 1 = m*1 + b ∧ 
                 0 = m*0 + b) :=
by sorry

end parallel_line_through_point_l1011_101186


namespace library_book_purchase_l1011_101100

/-- The library's book purchase problem -/
theorem library_book_purchase :
  let total_spent : ℕ := 4500
  let total_books : ℕ := 300
  let price_zhuangzi : ℕ := 10
  let price_confucius : ℕ := 20
  let price_mencius : ℕ := 15
  let price_laozi : ℕ := 28
  let price_sunzi : ℕ := 12
  ∀ (num_zhuangzi num_confucius num_mencius num_laozi num_sunzi : ℕ),
    num_zhuangzi + num_confucius + num_mencius + num_laozi + num_sunzi = total_books →
    num_zhuangzi * price_zhuangzi + num_confucius * price_confucius + 
    num_mencius * price_mencius + num_laozi * price_laozi + 
    num_sunzi * price_sunzi = total_spent →
    num_zhuangzi = num_confucius →
    num_sunzi = 4 * num_laozi + 15 →
    num_sunzi = 195 :=
by sorry

end library_book_purchase_l1011_101100


namespace students_surveyed_l1011_101128

theorem students_surveyed : ℕ :=
  let total_students : ℕ := sorry
  let french_speakers : ℕ := sorry
  let french_english_speakers : ℕ := 10
  let french_only_speakers : ℕ := 40

  have h1 : french_speakers = french_english_speakers + french_only_speakers := by sorry
  have h2 : french_speakers = 50 := by sorry
  have h3 : french_speakers = total_students / 4 := by sorry

  200

/- Proof omitted -/

end students_surveyed_l1011_101128


namespace solve_system_l1011_101190

theorem solve_system (a b : ℤ) 
  (eq1 : 2013 * a + 2015 * b = 2023)
  (eq2 : 2017 * a + 2019 * b = 2027) :
  a - b = -9 := by sorry

end solve_system_l1011_101190


namespace bus_empty_seats_after_second_stop_l1011_101103

/-- Represents the state of the bus at different stages --/
structure BusState where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Calculates the number of empty seats in the bus --/
def empty_seats (state : BusState) : ℕ :=
  state.total_seats - state.occupied_seats

/-- Updates the bus state after passenger movement --/
def update_state (state : BusState) (board : ℕ) (leave : ℕ) : BusState :=
  { total_seats := state.total_seats,
    occupied_seats := state.occupied_seats + board - leave }

theorem bus_empty_seats_after_second_stop :
  let initial_state : BusState := { total_seats := 23 * 4, occupied_seats := 16 }
  let first_stop := update_state initial_state 15 3
  let second_stop := update_state first_stop 17 10
  empty_seats second_stop = 57 := by sorry


end bus_empty_seats_after_second_stop_l1011_101103


namespace ecommerce_sales_analysis_l1011_101173

/-- Represents the sales model for an e-commerce platform. -/
structure SalesModel where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ

/-- Calculates the daily sales volume for a given price. -/
def daily_sales (model : SalesModel) (price : ℝ) : ℝ :=
  model.initial_sales + model.price_sensitivity * (model.initial_price - price)

/-- Calculates the daily profit for a given price. -/
def daily_profit (model : SalesModel) (price : ℝ) : ℝ :=
  (price - model.cost_price) * daily_sales model price

/-- The e-commerce platform's sales model. -/
def ecommerce_model : SalesModel := {
  cost_price := 40
  initial_price := 60
  initial_sales := 20
  price_sensitivity := 2
}

/-- Xiao Ming's store price. -/
def xiaoming_price : ℝ := 62.5

theorem ecommerce_sales_analysis 
  (h1 : daily_sales ecommerce_model 45 = 50)
  (h2 : ∃ x, x ≥ 40 ∧ x < 60 ∧ daily_profit ecommerce_model x = daily_profit ecommerce_model 60 ∧
             ∀ y, y ≥ 40 ∧ y < 60 ∧ daily_profit ecommerce_model y = daily_profit ecommerce_model 60 → x ≤ y)
  (h3 : ∃ d : ℝ, d ≥ 0 ∧ d ≤ 1 ∧ xiaoming_price * (1 - d) ≤ 50 ∧
             ∀ e, e ≥ 0 ∧ e < d ∧ xiaoming_price * (1 - e) ≤ 50 → False) :
  (daily_sales ecommerce_model 45 = 50) ∧
  (∃ x, x = 50 ∧ daily_profit ecommerce_model x = daily_profit ecommerce_model 60 ∧
        ∀ y, y ≥ 40 ∧ y < 60 ∧ daily_profit ecommerce_model y = daily_profit ecommerce_model 60 → x ≤ y) ∧
  (∃ d : ℝ, d = 0.2 ∧ xiaoming_price * (1 - d) ≤ 50 ∧
            ∀ e, e ≥ 0 ∧ e < d ∧ xiaoming_price * (1 - e) ≤ 50 → False) := by
  sorry

end ecommerce_sales_analysis_l1011_101173


namespace problem_statement_l1011_101122

theorem problem_statement (a b : ℝ) (ha : a ≠ b) 
  (ha_eq : a^2 - 13*a + 1 = 0) (hb_eq : b^2 - 13*b + 1 = 0) :
  b / (1 + b) + (a^2 + a) / (a^2 + 2*a + 1) = 1 := by
  sorry

end problem_statement_l1011_101122


namespace thirty_times_multiple_of_every_integer_l1011_101130

theorem thirty_times_multiple_of_every_integer (n : ℤ) :
  (∀ m : ℤ, ∃ k : ℤ, n = 30 * k * m) → n = 0 := by
  sorry

end thirty_times_multiple_of_every_integer_l1011_101130


namespace total_wheels_l1011_101121

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of adults riding bicycles -/
def adults_on_bicycles : ℕ := 6

/-- The number of children riding tricycles -/
def children_on_tricycles : ℕ := 15

/-- The total number of wheels Dimitri saw at the park -/
theorem total_wheels : 
  bicycle_wheels * adults_on_bicycles + tricycle_wheels * children_on_tricycles = 57 := by
  sorry

end total_wheels_l1011_101121


namespace sum_of_reciprocals_l1011_101153

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 14) (h2 : x * y = 45) :
  1 / x + 1 / y = 14 / 45 := by
sorry

end sum_of_reciprocals_l1011_101153


namespace work_hours_theorem_l1011_101117

/-- Calculates the total hours worked given the number of days and hours per day -/
def total_hours (days : ℝ) (hours_per_day : ℝ) : ℝ :=
  days * hours_per_day

/-- Proves that working 2 hours per day for 4 days results in 8 total hours -/
theorem work_hours_theorem :
  let days : ℝ := 4
  let hours_per_day : ℝ := 2
  total_hours days hours_per_day = 8 := by
  sorry

end work_hours_theorem_l1011_101117
