import Mathlib

namespace hindi_speakers_count_l686_68671

/-- Represents the number of students who can speak a certain number of languages -/
structure LanguageSpeakers where
  total : ℕ  -- Total number of students in the class
  gujarati : ℕ  -- Number of students who can speak Gujarati
  marathi : ℕ  -- Number of students who can speak Marathi
  twoLanguages : ℕ  -- Number of students who can speak two languages
  allThree : ℕ  -- Number of students who can speak all three languages

/-- Calculates the number of Hindi speakers given the language distribution in the class -/
def numHindiSpeakers (ls : LanguageSpeakers) : ℕ :=
  ls.total - (ls.gujarati + ls.marathi - ls.twoLanguages + ls.allThree)

/-- Theorem stating that the number of Hindi speakers is 10 given the problem conditions -/
theorem hindi_speakers_count (ls : LanguageSpeakers) 
  (h_total : ls.total = 22)
  (h_gujarati : ls.gujarati = 6)
  (h_marathi : ls.marathi = 6)
  (h_two : ls.twoLanguages = 2)
  (h_all : ls.allThree = 1) :
  numHindiSpeakers ls = 10 := by
  sorry


end hindi_speakers_count_l686_68671


namespace max_square_in_unit_triangle_l686_68620

/-- A triangle with base and height both equal to √2 maximizes the area of the inscribed square among all unit-area triangles. -/
theorem max_square_in_unit_triangle :
  ∀ (base height : ℝ) (square_side : ℝ),
    base > 0 → height > 0 → square_side > 0 →
    (1/2) * base * height = 1 →
    square_side^2 ≤ 1/2 →
    square_side^2 ≤ (base * height) / (base + height)^2 →
    square_side^2 ≤ 1/2 :=
by sorry

end max_square_in_unit_triangle_l686_68620


namespace inequality_proof_l686_68660

theorem inequality_proof (a b c : ℝ) : 
  a = 4/5 → b = Real.sin (2/3) → c = Real.cos (1/3) → b < a ∧ a < c := by
  sorry

end inequality_proof_l686_68660


namespace expression_factorization_l686_68689

theorem expression_factorization (x : ℝ) :
  (4 * x^3 - 64 * x^2 + 52) - (-3 * x^3 - 2 * x^2 + 52) = x^2 * (7 * x - 62) := by
  sorry

end expression_factorization_l686_68689


namespace athlete_distance_l686_68625

/-- Proves that an athlete running at 18 km/h for 40 seconds covers 200 meters -/
theorem athlete_distance (speed_kmh : ℝ) (time_s : ℝ) (distance_m : ℝ) : 
  speed_kmh = 18 → time_s = 40 → distance_m = speed_kmh * (1000 / 3600) * time_s → distance_m = 200 := by
  sorry

#check athlete_distance

end athlete_distance_l686_68625


namespace sqrt_problems_l686_68682

-- Define the arithmetic square root
noncomputable def arithmeticSqrt (x : ℝ) : ℝ := Real.sqrt x

-- Define the square root function that returns a set
def squareRoot (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

theorem sqrt_problems :
  (∀ x : ℝ, x > 0 → arithmeticSqrt x ≥ 0) ∧
  (squareRoot 81 = {9, -9}) ∧
  (|2 - Real.sqrt 5| = Real.sqrt 5 - 2) ∧
  (Real.sqrt (4/121) = 2/11) ∧
  (2 * Real.sqrt 3 - 5 * Real.sqrt 3 = -3 * Real.sqrt 3) :=
by sorry

end sqrt_problems_l686_68682


namespace ellipse_eccentricity_specific_ellipse_eccentricity_l686_68684

/-- The eccentricity of an ellipse with equation x²/a² + y²/b² = 1 is √(1 - b²/a²) -/
theorem ellipse_eccentricity (a b : ℝ) (h : 0 < b ∧ b < a) :
  let e := Real.sqrt (1 - b^2 / a^2)
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  e = Real.sqrt (1 - b^2 / a^2) :=
sorry

/-- The eccentricity of the ellipse x²/9 + y² = 1 is 2√2/3 -/
theorem specific_ellipse_eccentricity :
  let e := Real.sqrt (1 - 1^2 / 3^2)
  (∀ x y : ℝ, x^2 / 9 + y^2 = 1) →
  e = 2 * Real.sqrt 2 / 3 :=
sorry

end ellipse_eccentricity_specific_ellipse_eccentricity_l686_68684


namespace parabola_directrix_l686_68627

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := x^2 = 4*y

/-- The directrix equation -/
def directrix_eq (y : ℝ) : Prop := y = -1

/-- Theorem: The directrix of the parabola x^2 = 4y is y = -1 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_eq x y → ∃ d : ℝ, directrix_eq d ∧ 
  (∀ p : ℝ × ℝ, p.1^2 = 4*p.2 → (p.1^2 + (p.2 - d)^2) = (p.2 - d)^2) :=
sorry

end parabola_directrix_l686_68627


namespace distance_to_directrix_l686_68641

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ

/-- Point type representing (x, y) coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Distance from point A to directrix of parabola C -/
theorem distance_to_directrix 
  (C : Parabola) 
  (A : Point) 
  (h1 : A.y^2 = 2 * C.p * A.x) 
  (h2 : A.x = 1) 
  (h3 : A.y = Real.sqrt 5) : 
  A.x + C.p / 2 = 9 / 4 := by
  sorry

#check distance_to_directrix

end distance_to_directrix_l686_68641


namespace triangle_max_area_l686_68658

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  3 * b * Real.sin C - 5 * c * Real.sin B * Real.cos A = 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ 2) :=
by sorry

end triangle_max_area_l686_68658


namespace preimage_of_two_zero_l686_68637

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (1, 1) is the preimage of (2, 0) under f -/
theorem preimage_of_two_zero :
  f (1, 1) = (2, 0) ∧ ∀ p : ℝ × ℝ, f p = (2, 0) → p = (1, 1) := by
  sorry

end preimage_of_two_zero_l686_68637


namespace increasing_function_composition_l686_68646

theorem increasing_function_composition (f : ℝ → ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x > y → f x - f y > x - y) →
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x > y → f (x^2) - f (y^2) > x^6 - y^6) →
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x > y → f (x^3) - f (y^3) > (Real.sqrt 3 / 2) * (x^6 - y^6)) :=
by sorry

end increasing_function_composition_l686_68646


namespace inequality_solution_l686_68686

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x ↔ x = 1) :=
by sorry

end inequality_solution_l686_68686


namespace harkamal_fruit_payment_l686_68635

/-- Calculates the total amount Harkamal had to pay for fruits with given quantities, prices, discount, and tax rates. -/
def calculate_total_payment (grape_kg : ℝ) (grape_price : ℝ) (mango_kg : ℝ) (mango_price : ℝ)
                            (apple_kg : ℝ) (apple_price : ℝ) (orange_kg : ℝ) (orange_price : ℝ)
                            (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let grape_total := grape_kg * grape_price
  let mango_total := mango_kg * mango_price
  let apple_total := apple_kg * apple_price
  let orange_total := orange_kg * orange_price
  let total_before_discount := grape_total + mango_total + apple_total + orange_total
  let discount := discount_rate * (grape_total + apple_total)
  let price_after_discount := total_before_discount - discount
  let tax := tax_rate * price_after_discount
  price_after_discount + tax

/-- Theorem stating that the total payment for Harkamal's fruit purchase is $1507.32. -/
theorem harkamal_fruit_payment :
  calculate_total_payment 9 70 9 55 5 40 6 30 0.1 0.06 = 1507.32 := by
  sorry

end harkamal_fruit_payment_l686_68635


namespace no_prime_p_and_p6_plus_6_prime_l686_68610

theorem no_prime_p_and_p6_plus_6_prime :
  ¬ ∃ (p : ℕ), Nat.Prime p ∧ Nat.Prime (p^6 + 6) := by
  sorry

end no_prime_p_and_p6_plus_6_prime_l686_68610


namespace lens_curve_properties_l686_68642

/-- A lens-shaped curve consisting of two equal circular arcs -/
structure LensCurve where
  radius : ℝ
  arc_angle : ℝ
  h_positive_radius : 0 < radius
  h_arc_angle : arc_angle = 2 * Real.pi / 3

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  h_positive_side : 0 < side_length

/-- Predicate to check if a curve is closed and non-self-intersecting -/
def is_closed_non_self_intersecting (curve : Type) : Prop := sorry

/-- Predicate to check if a curve is different from a circle -/
def is_not_circle (curve : Type) : Prop := sorry

/-- Predicate to check if a triangle can be moved inside a curve with vertices tracing the curve -/
def can_move_triangle_inside (curve : Type) (triangle : Type) : Prop := sorry

theorem lens_curve_properties (l : LensCurve) (t : EquilateralTriangle) 
  (h : l.radius = t.side_length) : 
  is_closed_non_self_intersecting LensCurve ∧ 
  is_not_circle LensCurve ∧ 
  can_move_triangle_inside LensCurve EquilateralTriangle := by
  sorry

end lens_curve_properties_l686_68642


namespace binomial_plus_four_l686_68677

theorem binomial_plus_four : (Nat.choose 18 17) + 4 = 22 := by
  sorry

end binomial_plus_four_l686_68677


namespace sin_double_minus_cos_half_squared_l686_68648

theorem sin_double_minus_cos_half_squared 
  (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.sin (Real.pi - α) = 4 / 5) : 
  Real.sin (2 * α) - Real.cos (α / 2) ^ 2 = 4 / 25 := by
sorry

end sin_double_minus_cos_half_squared_l686_68648


namespace ending_number_is_48_l686_68632

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def sequence_start : ℕ := 10

def sequence_length : ℕ := 13

theorem ending_number_is_48 :
  ∃ (seq : ℕ → ℕ),
    (∀ i, seq i ≥ sequence_start) ∧
    (∀ i, is_divisible_by_3 (seq i)) ∧
    (∀ i j, i < j → seq i < seq j) ∧
    (seq 0 = (sequence_start + 2)) ∧
    (seq (sequence_length - 1) = 48) :=
sorry

end ending_number_is_48_l686_68632


namespace a_zero_sufficient_not_necessary_l686_68624

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x² + a(b+1)x + a + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*(b+1)*x + a + b

/-- "a = 0" is a sufficient but not necessary condition for "f is an even function" -/
theorem a_zero_sufficient_not_necessary (a b : ℝ) :
  (a = 0 → IsEven (f a b)) ∧ ¬(IsEven (f a b) → a = 0) := by
  sorry

end a_zero_sufficient_not_necessary_l686_68624


namespace h_of_3_eq_3_l686_68638

-- Define the function h
noncomputable def h : ℝ → ℝ := fun x => 
  if x = 1 then 0  -- Handle the case when x = 1 separately
  else ((x + 1) * (x^2 + 1) * (x^3 + 1) * (x^4 + 1) * (x^5 + 1) * (x^6 + 1) * (x^7 + 1) * (x^8 + 1) * (x^9 + 1) - 1) / (x^26 - 1)

-- Theorem statement
theorem h_of_3_eq_3 : h 3 = 3 := by sorry

end h_of_3_eq_3_l686_68638


namespace train_speed_ratio_l686_68606

/-- Given two trains running in opposite directions, prove that their speed ratio is 39:5 -/
theorem train_speed_ratio :
  ∀ (v₁ v₂ : ℝ), v₁ > 0 → v₂ > 0 →
  ∃ (l₁ l₂ : ℝ), l₁ > 0 ∧ l₂ > 0 ∧
  (l₁ / v₁ = 27) ∧ (l₂ / v₂ = 17) ∧ ((l₁ + l₂) / (v₁ + v₂) = 22) →
  v₁ / v₂ = 39 / 5 := by
sorry

end train_speed_ratio_l686_68606


namespace equal_area_triangle_square_l686_68673

/-- A square with vertices O, S, U, V -/
structure Square (O S U V : ℝ × ℝ) : Prop where
  is_square : true  -- We assume OSUV is a square without proving it

/-- The area of a triangle given three points -/
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

/-- The area of a square given its side length -/
def square_area (side : ℝ) : ℝ := side * side

theorem equal_area_triangle_square 
  (O S U V W : ℝ × ℝ) 
  (h_square : Square O S U V)
  (h_O : O = (0, 0))
  (h_U : U = (3, 3))
  (h_W : W = (3, 9)) : 
  triangle_area S V W = square_area 3 := by
  sorry

#check equal_area_triangle_square

end equal_area_triangle_square_l686_68673


namespace min_sum_squares_l686_68683

/-- B-neighborhood of A is defined as the solution set of |x-A| < B -/
def neighborhood (A B : ℝ) := {x : ℝ | |x - A| < B}

theorem min_sum_squares (a b : ℝ) : 
  neighborhood (a + b - 3) (a + b) = Set.Ioo (-3 : ℝ) 3 → 
  ∃ (min : ℝ), min = 9/2 ∧ ∀ x y : ℝ, x^2 + y^2 ≥ min := by
  sorry

end min_sum_squares_l686_68683


namespace misread_weight_l686_68617

/-- Proves that the misread weight is 56 kg given the conditions of the problem -/
theorem misread_weight (n : ℕ) (initial_avg correct_avg : ℝ) (correct_weight : ℝ) :
  n = 20 ∧ 
  initial_avg = 58.4 ∧ 
  correct_avg = 59 ∧ 
  correct_weight = 68 →
  ∃ x : ℝ, x = 56 ∧ n * correct_avg - n * initial_avg = correct_weight - x :=
by sorry

end misread_weight_l686_68617


namespace equal_differences_l686_68679

theorem equal_differences (x : Fin 102 → ℕ) 
  (h_increasing : ∀ i j : Fin 102, i < j → x i < x j)
  (h_upper_bound : ∀ i : Fin 102, x i < 255) :
  ∃ (S : Finset (Fin 101)) (d : ℕ), 
    S.card ≥ 26 ∧ ∀ i ∈ S, x (i + 1) - x i = d := by
  sorry

end equal_differences_l686_68679


namespace trig_identity_l686_68674

theorem trig_identity (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := by
  sorry

end trig_identity_l686_68674


namespace square_less_than_triple_l686_68675

theorem square_less_than_triple (n : ℕ) : n > 0 → (n^2 < 3*n ↔ n = 1 ∨ n = 2) := by
  sorry

end square_less_than_triple_l686_68675


namespace hyperbola_equilateral_triangle_l686_68653

/-- Hyperbola type representing xy = 1 -/
structure Hyperbola where
  C₁ : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.1 * p.2 = 1}
  C₂ : Set (ℝ × ℝ) := {p | p.1 < 0 ∧ p.1 * p.2 = 1}

/-- Predicate to check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p q r : ℝ × ℝ) : Prop :=
  let d₁ := (p.1 - q.1)^2 + (p.2 - q.2)^2
  let d₂ := (q.1 - r.1)^2 + (q.2 - r.2)^2
  let d₃ := (r.1 - p.1)^2 + (r.2 - p.2)^2
  d₁ = d₂ ∧ d₂ = d₃

/-- Main theorem statement -/
theorem hyperbola_equilateral_triangle (h : Hyperbola) (p q r : ℝ × ℝ) 
  (hp : p = (-1, -1) ∧ p ∈ h.C₂)
  (hq : q ∈ h.C₁)
  (hr : r ∈ h.C₁)
  (heq : IsEquilateralTriangle p q r) :
  (¬ (p ∈ h.C₁ ∧ q ∈ h.C₁ ∧ r ∈ h.C₁)) ∧
  (q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ r = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) := by
  sorry

end hyperbola_equilateral_triangle_l686_68653


namespace min_apples_in_basket_l686_68657

theorem min_apples_in_basket (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 4 = 3) ∧ (x % 5 = 2) → x ≥ 67 :=
by sorry

end min_apples_in_basket_l686_68657


namespace spliced_wire_length_l686_68645

theorem spliced_wire_length 
  (num_pieces : ℕ) 
  (piece_length : ℝ) 
  (overlap : ℝ) 
  (h1 : num_pieces = 15) 
  (h2 : piece_length = 25) 
  (h3 : overlap = 0.5) : 
  (num_pieces * piece_length - (num_pieces - 1) * overlap) / 100 = 3.68 := by
sorry

end spliced_wire_length_l686_68645


namespace multiply_algebraic_expressions_l686_68621

theorem multiply_algebraic_expressions (x y : ℝ) :
  6 * x * y^3 * (-1/2 * x^3 * y^2) = -3 * x^4 * y^5 := by
  sorry

end multiply_algebraic_expressions_l686_68621


namespace candy_necklaces_remaining_l686_68609

/-- Proves that given 9 packs of candy necklaces with 8 necklaces in each pack,
    if 4 packs are opened, then at least 40 candy necklaces remain unopened. -/
theorem candy_necklaces_remaining (total_packs : ℕ) (necklaces_per_pack : ℕ) (opened_packs : ℕ) :
  total_packs = 9 →
  necklaces_per_pack = 8 →
  opened_packs = 4 →
  (total_packs - opened_packs) * necklaces_per_pack ≥ 40 := by
  sorry

end candy_necklaces_remaining_l686_68609


namespace original_number_proof_l686_68659

theorem original_number_proof (x : ℚ) :
  1 + (1 / x) = 5 / 2 → x = 2 / 3 := by
  sorry

end original_number_proof_l686_68659


namespace algebraic_expression_value_l686_68626

theorem algebraic_expression_value (x y : ℝ) 
  (sum_eq : x + y = 2) 
  (diff_eq : x - y = 4) : 
  1 + x^2 - y^2 = 9 := by
  sorry

end algebraic_expression_value_l686_68626


namespace last_digit_of_one_over_three_to_ninth_l686_68631

theorem last_digit_of_one_over_three_to_ninth (n : ℕ) : n = 3^9 → (1000000000 / n) % 10 = 7 := by
  sorry

end last_digit_of_one_over_three_to_ninth_l686_68631


namespace fish_remaining_l686_68601

theorem fish_remaining (initial : ℝ) (moved : ℝ) :
  initial ≥ moved →
  initial - moved = initial - moved :=
by sorry

end fish_remaining_l686_68601


namespace smallest_number_divisibility_l686_68600

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 125 * k)) ∧ 
  (∃ k₁ k₂ k₃ : ℕ, (n + 7) = 125 * k₁ ∧ (n + 7) = 11 * k₂ ∧ (n + 7) = 24 * k₃) ∧
  (∃ k : ℕ, n = 8 * k) ∧
  (n + 7 = 257) → 
  n = 250 := by
sorry

end smallest_number_divisibility_l686_68600


namespace remainder_13_pow_2011_mod_100_l686_68698

theorem remainder_13_pow_2011_mod_100 : 13^2011 % 100 = 37 := by
  sorry

end remainder_13_pow_2011_mod_100_l686_68698


namespace indifferent_passengers_adjacent_probability_l686_68663

/-- The number of seats on each sofa -/
def seats_per_sofa : ℕ := 5

/-- The total number of passengers -/
def total_passengers : ℕ := 10

/-- The number of passengers who prefer to sit facing the locomotive -/
def facing_passengers : ℕ := 4

/-- The number of passengers who prefer to sit with their backs to the locomotive -/
def back_passengers : ℕ := 3

/-- The number of passengers who do not care where they sit -/
def indifferent_passengers : ℕ := 3

/-- The probability that two of the three indifferent passengers sit next to each other -/
theorem indifferent_passengers_adjacent_probability :
  (seats_per_sofa = 5) →
  (total_passengers = 10) →
  (facing_passengers = 4) →
  (back_passengers = 3) →
  (indifferent_passengers = 3) →
  (Nat.factorial seats_per_sofa * Nat.factorial 3 * 2 * 4) / 
  (3 * Nat.factorial seats_per_sofa * Nat.factorial seats_per_sofa) = 2 / 15 := by
  sorry


end indifferent_passengers_adjacent_probability_l686_68663


namespace max_value_implies_m_l686_68676

noncomputable def f (x m : ℝ) : ℝ :=
  2 * (Real.cos x) * (Real.cos x) + Real.sqrt 3 * Real.sin (2 * x) + m

theorem max_value_implies_m (h : ∀ x ∈ Set.Icc 0 (Real.pi / 6), f x 1 ≤ 4) :
  ∃ x ∈ Set.Icc 0 (Real.pi / 6), f x 1 = 4 :=
sorry

end max_value_implies_m_l686_68676


namespace simplify_and_evaluate_l686_68623

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 5) :
  (a - 1)^2 - 2*a*(a - 1) = -4 := by
  sorry

end simplify_and_evaluate_l686_68623


namespace sum_of_cubes_equals_ten_squared_l686_68614

theorem sum_of_cubes_equals_ten_squared (h1 : 1 + 2 + 3 + 4 = 10) 
  (h2 : ∃ n : ℕ, 1^3 + 2^3 + 3^3 + 4^3 = 10^n) : 
  ∃ n : ℕ, 1^3 + 2^3 + 3^3 + 4^3 = 10^n ∧ n = 2 := by
  sorry

end sum_of_cubes_equals_ten_squared_l686_68614


namespace sum_of_divisor_and_quotient_l686_68696

/-- Given a valid vertical division, prove that the sum of the divisor and quotient is 723. -/
theorem sum_of_divisor_and_quotient : 
  ∀ (D Q : ℕ), 
  (D = 581) →  -- Divisor condition
  (Q = 142) →  -- Quotient condition
  (D + Q = 723) :=
by
  sorry

end sum_of_divisor_and_quotient_l686_68696


namespace eleventh_number_is_137_l686_68654

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits add up to 11 -/
def nth_number_with_digit_sum_11 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 11th number in the sequence is 137 -/
theorem eleventh_number_is_137 : nth_number_with_digit_sum_11 11 = 137 := by sorry

end eleventh_number_is_137_l686_68654


namespace addition_equality_l686_68628

theorem addition_equality : 731 + 672 = 1403 := by
  sorry

end addition_equality_l686_68628


namespace range_of_m_l686_68604

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3) ∧
  (∃ x : ℝ, x > 2 * m^2 - 3 ∧ (x ≤ -1 ∨ x ≥ 4)) →
  -1 ≤ m ∧ m ≤ 1 :=
by sorry

end range_of_m_l686_68604


namespace number_divisibility_problem_l686_68656

theorem number_divisibility_problem : 
  ∃ x : ℚ, (x / 3) * 12 = 9 ∧ x = 9 / 4 := by
  sorry

end number_divisibility_problem_l686_68656


namespace roses_distribution_l686_68655

def total_money : ℕ := 300
def jenna_price : ℕ := 2
def imma_price : ℕ := 3
def ravi_price : ℕ := 4
def leila_price : ℕ := 5

def jenna_budget : ℕ := 100
def imma_budget : ℕ := 100
def ravi_budget : ℕ := 50
def leila_budget : ℕ := 50

def jenna_fraction : ℚ := 1/3
def imma_fraction : ℚ := 1/4
def ravi_fraction : ℚ := 1/6

theorem roses_distribution (jenna_roses imma_roses ravi_roses leila_roses : ℕ) :
  jenna_roses = ⌊(jenna_fraction * (jenna_budget / jenna_price : ℚ))⌋ ∧
  imma_roses = ⌊(imma_fraction * (imma_budget / imma_price : ℚ))⌋ ∧
  ravi_roses = ⌊(ravi_fraction * (ravi_budget / ravi_price : ℚ))⌋ ∧
  leila_roses = leila_budget / leila_price →
  jenna_roses + imma_roses + ravi_roses + leila_roses = 36 := by
  sorry

end roses_distribution_l686_68655


namespace rickshaw_charge_calculation_l686_68667

/-- Rickshaw charge calculation -/
theorem rickshaw_charge_calculation 
  (initial_charge : ℝ) 
  (additional_charge : ℝ) 
  (total_distance : ℝ) 
  (total_charge : ℝ) :
  initial_charge = 13.5 →
  additional_charge = 2.5 →
  total_distance = 13 →
  total_charge = 103.5 →
  initial_charge + additional_charge * (total_distance - 1) = total_charge :=
by sorry

end rickshaw_charge_calculation_l686_68667


namespace ellipse_eccentricity_l686_68640

-- Define the ellipse
def ellipse (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 2) + y^2 / (k + 1) = 1

-- Define the foci
structure Foci (k : ℝ) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

-- Define the chord AB
structure Chord (k : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ

-- Define the eccentricity
def eccentricity (k : ℝ) : ℝ := sorry

theorem ellipse_eccentricity 
  (k : ℝ) 
  (hk : k > -1) 
  (f : Foci k)
  (c : Chord k)
  (hF₁ : c.A.1 = f.F₁.1 ∧ c.A.2 = f.F₁.2) -- Chord AB passes through F₁
  (hPerimeter : Real.sqrt ((c.A.1 - f.F₂.1)^2 + (c.A.2 - f.F₂.2)^2) + 
                Real.sqrt ((c.B.1 - f.F₂.1)^2 + (c.B.2 - f.F₂.2)^2) + 
                Real.sqrt ((c.A.1 - c.B.1)^2 + (c.A.2 - c.B.2)^2) = 8) -- Perimeter of ABF₂ is 8
  : eccentricity k = 1/2 := by
  sorry


end ellipse_eccentricity_l686_68640


namespace dodecagon_diagonals_l686_68633

/-- The number of sides in a dodecagon -/
def n : ℕ := 12

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a regular dodecagon is 54 -/
theorem dodecagon_diagonals : num_diagonals n = 54 := by
  sorry

end dodecagon_diagonals_l686_68633


namespace infinite_triangles_with_side_ten_l686_68680

/-- A function that checks if three positive integers can form a triangle -/
def can_form_triangle (a b c : ℕ+) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating that there are infinitely many triangles with sides x, y, and 10 -/
theorem infinite_triangles_with_side_ten :
  ∀ n : ℕ, ∃ x y : ℕ+, x > n ∧ y > n ∧ can_form_triangle x y 10 :=
sorry

end infinite_triangles_with_side_ten_l686_68680


namespace solution_symmetry_l686_68605

theorem solution_symmetry (x y : ℝ) : 
  ((x - y) * (x^2 - y^2) = 160 ∧ (x + y) * (x^2 + y^2) = 580) →
  ((3 - 7) * (3^2 - 7^2) = 160 ∧ (3 + 7) * (3^2 + 7^2) = 580) →
  ((7 - 3) * (7^2 - 3^2) = 160 ∧ (7 + 3) * (7^2 + 3^2) = 580) := by
sorry

end solution_symmetry_l686_68605


namespace fraction_to_decimal_l686_68678

theorem fraction_to_decimal : (3 : ℚ) / 24 = 0.125 := by
  sorry

end fraction_to_decimal_l686_68678


namespace pokemon_cards_total_l686_68652

theorem pokemon_cards_total (jenny : ℕ) (orlando : ℕ) (richard : ℕ) : 
  jenny = 6 →
  orlando = jenny + 2 →
  richard = 3 * orlando →
  jenny + orlando + richard = 38 := by
sorry

end pokemon_cards_total_l686_68652


namespace jimmy_has_more_sheets_l686_68681

/-- Represents the number of sheets each person has -/
structure Sheets where
  jimmy : ℕ
  tommy : ℕ
  ashton : ℕ

/-- The initial state of sheet distribution -/
def initial : Sheets where
  jimmy := 58
  tommy := 58 + 25
  ashton := 85

/-- The state after Ashton gives sheets to Jimmy -/
def final (s : Sheets) : Sheets where
  jimmy := s.jimmy + s.ashton
  tommy := s.tommy
  ashton := 0

/-- Theorem stating that Jimmy will have 60 more sheets than Tommy after receiving sheets from Ashton -/
theorem jimmy_has_more_sheets : (final initial).jimmy - (final initial).tommy = 60 := by
  sorry

end jimmy_has_more_sheets_l686_68681


namespace pufferfish_count_swordfish_to_pufferfish_ratio_total_fish_count_l686_68651

/-- The number of pufferfish in an aquarium exhibit -/
def num_pufferfish : ℕ := 15

/-- The number of swordfish in the aquarium exhibit -/
def num_swordfish : ℕ := 5 * num_pufferfish

/-- The total number of fish in the aquarium exhibit -/
def total_fish : ℕ := 90

/-- Theorem stating that the number of pufferfish is 15 -/
theorem pufferfish_count : num_pufferfish = 15 := by sorry

/-- Theorem stating that the number of swordfish is five times the number of pufferfish -/
theorem swordfish_to_pufferfish_ratio : num_swordfish = 5 * num_pufferfish := by sorry

/-- Theorem stating that the total number of fish is 90 -/
theorem total_fish_count : total_fish = num_swordfish + num_pufferfish := by sorry

end pufferfish_count_swordfish_to_pufferfish_ratio_total_fish_count_l686_68651


namespace range_of_M_l686_68693

theorem range_of_M (x y z : ℝ) 
  (h1 : x + y + z = 30)
  (h2 : 3 * x + y - z = 50)
  (h3 : x ≥ 0)
  (h4 : y ≥ 0)
  (h5 : z ≥ 0) :
  120 ≤ 5 * x + 4 * y + 2 * z ∧ 5 * x + 4 * y + 2 * z ≤ 130 :=
by sorry

end range_of_M_l686_68693


namespace second_bus_students_l686_68672

theorem second_bus_students (first_bus : ℕ) (second_bus : ℕ) : 
  first_bus = 38 →
  second_bus - 4 = (first_bus + 4) + 2 →
  second_bus = 44 := by
sorry

end second_bus_students_l686_68672


namespace not_p_sufficient_not_necessary_for_q_l686_68608

theorem not_p_sufficient_not_necessary_for_q :
  (∃ a : ℝ, (a < -1 → ∀ x > 0, a ≤ (x^2 + 1) / x) ∧
   (∀ x > 0, a ≤ (x^2 + 1) / x → ¬(a < -1))) :=
sorry

end not_p_sufficient_not_necessary_for_q_l686_68608


namespace root_zero_implies_a_half_l686_68695

theorem root_zero_implies_a_half (a : ℝ) : 
  (∀ x : ℝ, x^2 + x + 2*a - 1 = 0 → x = 0 ∨ x ≠ 0) →
  (0^2 + 0 + 2*a - 1 = 0) →
  a = 1/2 := by
sorry

end root_zero_implies_a_half_l686_68695


namespace unique_three_digit_sum_l686_68629

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Theorem stating that 198 is the only three-digit number equal to 11 times the sum of its digits -/
theorem unique_three_digit_sum : ∃! n : ℕ, isThreeDigit n ∧ n = 11 * sumOfDigits n := by
  sorry

end unique_three_digit_sum_l686_68629


namespace exponential_sum_conjugate_l686_68690

theorem exponential_sum_conjugate (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = -1/3 + 5/8 * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = -1/3 - 5/8 * Complex.I :=
by sorry

end exponential_sum_conjugate_l686_68690


namespace lego_volume_proof_l686_68603

/-- The number of rows of Legos -/
def num_rows : ℕ := 7

/-- The number of columns of Legos -/
def num_columns : ℕ := 5

/-- The number of layers of Legos -/
def num_layers : ℕ := 3

/-- The length of a single Lego in centimeters -/
def lego_length : ℝ := 1

/-- The width of a single Lego in centimeters -/
def lego_width : ℝ := 1

/-- The height of a single Lego in centimeters -/
def lego_height : ℝ := 1

/-- The total volume of stacked Legos in cubic centimeters -/
def total_volume : ℝ := num_rows * num_columns * num_layers * lego_length * lego_width * lego_height

theorem lego_volume_proof : total_volume = 105 := by
  sorry

end lego_volume_proof_l686_68603


namespace joans_clothing_expenditure_l686_68613

/-- The total amount Joan spent on clothing --/
def total_spent (shorts jacket shirt shoes hat belt : ℝ)
  (jacket_discount shirt_discount : ℝ) (shoes_coupon : ℝ) : ℝ :=
  shorts + (jacket * (1 - jacket_discount)) + (shirt * shirt_discount) +
  (shoes - shoes_coupon) + hat + belt

/-- Theorem stating the total amount Joan spent on clothing --/
theorem joans_clothing_expenditure :
  let shorts : ℝ := 15
  let jacket : ℝ := 14.82
  let shirt : ℝ := 12.51
  let shoes : ℝ := 21.67
  let hat : ℝ := 8.75
  let belt : ℝ := 6.34
  let jacket_discount : ℝ := 0.1  -- 10% discount on jacket
  let shirt_discount : ℝ := 0.5   -- half price for shirt
  let shoes_coupon : ℝ := 3       -- $3 off coupon for shoes
  total_spent shorts jacket shirt shoes hat belt jacket_discount shirt_discount shoes_coupon = 68.353 := by
  sorry


end joans_clothing_expenditure_l686_68613


namespace quadratic_minimum_l686_68622

/-- The quadratic function f(x) = x^2 - 14x + 45 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 45

theorem quadratic_minimum :
  (∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x) ∧
  (∃ (x : ℝ), f x = -4) ∧
  (∀ (y : ℝ), f y ≥ -4) ∧
  f 7 = -4 :=
sorry

end quadratic_minimum_l686_68622


namespace ceiling_sqrt_162_l686_68616

theorem ceiling_sqrt_162 : ⌈Real.sqrt 162⌉ = 13 := by sorry

end ceiling_sqrt_162_l686_68616


namespace cube_root_of_64_l686_68647

theorem cube_root_of_64 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 64) : x = 4 := by
  sorry

end cube_root_of_64_l686_68647


namespace books_in_box_l686_68611

theorem books_in_box (total : ℕ) (difference : ℕ) (books_a : ℕ) (books_b : ℕ) : 
  total = 20 → 
  difference = 4 → 
  books_a + books_b = total → 
  books_a = books_b + difference → 
  books_a = 12 := by
sorry

end books_in_box_l686_68611


namespace proposition_q_undetermined_l686_68607

theorem proposition_q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) := by sorry

end proposition_q_undetermined_l686_68607


namespace speed_conversion_l686_68644

/-- Conversion factor from m/s to km/h -/
def mps_to_kmh : ℝ := 3.6

/-- The given speed in km/h -/
def given_speed_kmh : ℝ := 1.1076923076923078

/-- The speed in m/s to be proven -/
def speed_mps : ℝ := 0.3076923076923077

theorem speed_conversion :
  speed_mps * mps_to_kmh = given_speed_kmh := by
  sorry

end speed_conversion_l686_68644


namespace rectangle_formation_count_l686_68634

/-- The number of ways to choose lines forming a rectangle with color constraints -/
def rectangleChoices (totalHorizontal totalVertical redHorizontal blueVertical : ℕ) : ℕ :=
  let horizontalChoices := (redHorizontal.choose 1 * (totalHorizontal - redHorizontal + 1).choose 1) +
                           redHorizontal.choose 2
  let verticalChoices := (blueVertical.choose 1 * (totalVertical - blueVertical + 1).choose 1) +
                         blueVertical.choose 2
  horizontalChoices * verticalChoices

/-- Theorem stating the number of ways to form a rectangle with given constraints -/
theorem rectangle_formation_count :
  rectangleChoices 6 5 3 2 = 84 := by
  sorry

end rectangle_formation_count_l686_68634


namespace shaded_area_proof_l686_68619

theorem shaded_area_proof (carpet_side : ℝ) (S T : ℝ) : 
  carpet_side = 9 →
  carpet_side / S = 3 →
  S / T = 3 →
  S * S + 8 * T * T = 17 :=
by sorry

end shaded_area_proof_l686_68619


namespace quadratic_inequality_range_l686_68643

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x - 1 ≤ 0) ↔ a ≤ -1 := by
  sorry

end quadratic_inequality_range_l686_68643


namespace fewest_tiles_required_l686_68639

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ :=
  d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ :=
  feet * 12

/-- The dimensions of a tile in inches -/
def tileDimensions : Dimensions :=
  { length := 2, width := 3 }

/-- The dimensions of the region in feet -/
def regionDimensionsFeet : Dimensions :=
  { length := 4, width := 6 }

/-- The dimensions of the region in inches -/
def regionDimensionsInches : Dimensions :=
  { length := feetToInches regionDimensionsFeet.length,
    width := feetToInches regionDimensionsFeet.width }

/-- Theorem: The fewest number of tiles required to cover the region is 576 -/
theorem fewest_tiles_required :
  (area regionDimensionsInches) / (area tileDimensions) = 576 := by
  sorry

end fewest_tiles_required_l686_68639


namespace height_equality_l686_68685

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  p : ℝ -- semiperimeter
  ha : ℝ -- height corresponding to side a

-- State the theorem
theorem height_equality (t : Triangle) : 
  t.ha = (2 * (t.p - t.a) * Real.cos (t.β / 2) * Real.cos (t.γ / 2)) / Real.cos (t.α / 2) ∧
  t.ha = (2 * (t.p - t.b) * Real.sin (t.β / 2) * Real.cos (t.γ / 2)) / Real.sin (t.α / 2) := by
  sorry

end height_equality_l686_68685


namespace cube_surface_area_l686_68630

/-- Given a cube with side perimeter 24 cm, its surface area is 216 cm² -/
theorem cube_surface_area (side_perimeter : ℝ) (h : side_perimeter = 24) :
  6 * (side_perimeter / 4)^2 = 216 := by
  sorry

end cube_surface_area_l686_68630


namespace smallest_whole_number_above_sum_l686_68618

theorem smallest_whole_number_above_sum : ∃ (n : ℕ), 
  (n : ℚ) > (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/5 : ℚ) + (6 + 1/6 : ℚ) ∧ 
  n = 19 ∧ 
  ∀ (m : ℕ), m < n → (m : ℚ) ≤ (3 + 1/3 : ℚ) + (4 + 1/4 : ℚ) + (5 + 1/5 : ℚ) + (6 + 1/6 : ℚ) :=
by sorry

#check smallest_whole_number_above_sum

end smallest_whole_number_above_sum_l686_68618


namespace cylinder_radius_l686_68661

theorem cylinder_radius (length width : Real) (h1 : length = 3 * Real.pi) (h2 : width = Real.pi) :
  ∃ (r : Real), (r = 3/2 ∨ r = 1/2) ∧ 
  (2 * Real.pi * r = length ∨ 2 * Real.pi * r = width) := by
  sorry

end cylinder_radius_l686_68661


namespace simplify_expression_l686_68664

theorem simplify_expression : 5000 * (5000^9) * 2^1000 = 5000^10 * 2^1000 := by
  sorry

end simplify_expression_l686_68664


namespace johns_walking_distance_l686_68615

/-- Represents the journey of John to his workplace -/
def Johns_Journey (total_distance : ℝ) (skateboard_speed : ℝ) (walking_speed : ℝ) (total_time : ℝ) : Prop :=
  ∃ (skateboard_distance : ℝ) (walking_distance : ℝ),
    skateboard_distance + walking_distance = total_distance ∧
    skateboard_distance / skateboard_speed + walking_distance / walking_speed = total_time ∧
    walking_distance = 5.0

theorem johns_walking_distance :
  Johns_Journey 10 10 6 (66/60) →
  ∃ (walking_distance : ℝ), walking_distance = 5.0 :=
by
  sorry


end johns_walking_distance_l686_68615


namespace integral_x_squared_zero_to_one_l686_68687

theorem integral_x_squared_zero_to_one :
  ∫ x in (0 : ℝ)..(1 : ℝ), x^2 = (1 : ℝ) / 3 := by
  sorry

end integral_x_squared_zero_to_one_l686_68687


namespace solution_set_when_a_neg_three_a_range_when_subset_condition_l686_68668

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x + a| + |x - 2|

-- Theorem 1
theorem solution_set_when_a_neg_three :
  {x : ℝ | f x (-3) ≥ 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

-- Theorem 2
theorem a_range_when_subset_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f x a ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by sorry

end solution_set_when_a_neg_three_a_range_when_subset_condition_l686_68668


namespace new_home_library_capacity_l686_68670

theorem new_home_library_capacity 
  (M : ℚ) -- Millicent's total number of books
  (H : ℚ) -- Harold's total number of books
  (harold_ratio : H = (1/2) * M) -- Harold has 1/2 as many books as Millicent
  (harold_brings : ℚ) -- Number of books Harold brings
  (millicent_brings : ℚ) -- Number of books Millicent brings
  (harold_brings_def : harold_brings = (1/3) * H) -- Harold brings 1/3 of his books
  (millicent_brings_def : millicent_brings = (1/2) * M) -- Millicent brings 1/2 of her books
  : harold_brings + millicent_brings = (2/3) * M := by
  sorry

end new_home_library_capacity_l686_68670


namespace difference_of_squares_75_25_l686_68692

theorem difference_of_squares_75_25 : 75^2 - 25^2 = 5000 := by
  sorry

end difference_of_squares_75_25_l686_68692


namespace david_more_than_zachary_pushup_difference_is_thirty_l686_68665

/-- The number of push-ups David did -/
def david_pushups : ℕ := 37

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 7

/-- David did more push-ups than Zachary -/
theorem david_more_than_zachary : david_pushups > zachary_pushups := by sorry

/-- The difference in push-ups between David and Zachary -/
def pushup_difference : ℕ := david_pushups - zachary_pushups

theorem pushup_difference_is_thirty : pushup_difference = 30 := by sorry

end david_more_than_zachary_pushup_difference_is_thirty_l686_68665


namespace arithmetic_sequence_sum_l686_68662

theorem arithmetic_sequence_sum : 
  ∀ (a₁ l d : ℤ) (n : ℕ),
    a₁ = -48 →
    l = 0 →
    d = 2 →
    n = 25 →
    l = a₁ + (n - 1) * d →
    (n : ℤ) * (a₁ + l) / 2 = -600 :=
by sorry

end arithmetic_sequence_sum_l686_68662


namespace shaded_sectors_ratio_l686_68602

/-- Given three semicircular protractors with radii 1, 3, and 5,
    whose centers coincide and diameters align,
    prove that the ratio of the areas of the shaded sectors is 48 : 40 : 15 -/
theorem shaded_sectors_ratio (r₁ r₂ r₃ : ℝ) (S_A S_B S_C : ℝ) :
  r₁ = 1 → r₂ = 3 → r₃ = 5 →
  S_A = (π / 10) * (r₃^2 - r₂^2) →
  S_B = (π / 6) * (r₂^2 - r₁^2) →
  S_C = (π / 2) * r₁^2 →
  ∃ (k : ℝ), k > 0 ∧ S_A = 48 * k ∧ S_B = 40 * k ∧ S_C = 15 * k :=
by sorry

end shaded_sectors_ratio_l686_68602


namespace quadratic_rational_root_parity_l686_68697

theorem quadratic_rational_root_parity (a b c : ℤ) (h_a : a ≠ 0) :
  (∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) →
  (Even b ∨ Even c) →
  ¬(Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end quadratic_rational_root_parity_l686_68697


namespace sqrt_x_plus_five_equals_two_l686_68699

theorem sqrt_x_plus_five_equals_two (x : ℝ) (h : x = -1) : Real.sqrt (x + 5) = 2 := by
  sorry

end sqrt_x_plus_five_equals_two_l686_68699


namespace function_solution_set_l686_68666

theorem function_solution_set (a : ℝ) : 
  (∀ x : ℝ, (|2*x - a| + a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3)) → a = 1 := by
  sorry

end function_solution_set_l686_68666


namespace pet_fee_calculation_l686_68688

-- Define the given constants
def daily_rate : ℚ := 125
def stay_duration_days : ℕ := 14
def service_fee_rate : ℚ := 0.2
def security_deposit_rate : ℚ := 0.5
def security_deposit : ℚ := 1110

-- Define the pet fee
def pet_fee : ℚ := 120

-- Theorem statement
theorem pet_fee_calculation :
  let base_cost := daily_rate * stay_duration_days
  let service_fee := service_fee_rate * base_cost
  let total_without_pet_fee := base_cost + service_fee
  let total_with_pet_fee := security_deposit / security_deposit_rate
  total_with_pet_fee - total_without_pet_fee = pet_fee := by
  sorry


end pet_fee_calculation_l686_68688


namespace complex_number_sum_l686_68636

theorem complex_number_sum (z : ℂ) : z = (2 + Complex.I) / (1 - 2 * Complex.I) → 
  ∃ (a b : ℝ), z = a + b * Complex.I ∧ a + b = 1 := by
  sorry

end complex_number_sum_l686_68636


namespace addition_preserves_inequality_l686_68691

theorem addition_preserves_inequality (a b c d : ℝ) : a < b → c < d → a + c < b + d := by
  sorry

end addition_preserves_inequality_l686_68691


namespace youngest_child_age_l686_68649

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem youngest_child_age (children : Fin 6 → ℕ) : 
  (∀ i : Fin 6, is_prime (children i)) →
  (∃ y : ℕ, children 0 = y ∧ 
            children 1 = y + 2 ∧
            children 2 = y + 6 ∧
            children 3 = y + 8 ∧
            children 4 = y + 12 ∧
            children 5 = y + 14) →
  children 0 = 5 :=
by sorry

end youngest_child_age_l686_68649


namespace three_times_first_minus_second_l686_68650

theorem three_times_first_minus_second (x y : ℕ) : 
  x + y = 48 → y = 17 → 3 * x - y = 76 := by
  sorry

end three_times_first_minus_second_l686_68650


namespace final_number_of_boys_l686_68694

/-- Given the initial number of boys and additional boys in a school, 
    prove that the final number of boys is the sum of these two numbers. -/
theorem final_number_of_boys 
  (initial_boys : ℕ) 
  (additional_boys : ℕ) : 
  initial_boys + additional_boys = initial_boys + additional_boys :=
by sorry

end final_number_of_boys_l686_68694


namespace min_garden_width_proof_l686_68669

/-- The minimum width of a rectangular garden satisfying the given conditions -/
def min_garden_width : ℝ := 4

/-- The length of the garden in terms of its width -/
def garden_length (w : ℝ) : ℝ := w + 20

/-- The area of the garden in terms of its width -/
def garden_area (w : ℝ) : ℝ := w * garden_length w

theorem min_garden_width_proof :
  (∀ w : ℝ, w > 0 → garden_area w ≥ 120 → w ≥ min_garden_width) ∧
  garden_area min_garden_width ≥ 120 :=
sorry

end min_garden_width_proof_l686_68669


namespace probability_of_event_A_is_half_events_A_and_C_mutually_exclusive_l686_68612

/-- Represents the labels on the balls -/
inductive Label : Type
  | one : Label
  | two : Label
  | three : Label

/-- Represents a pair of drawn balls -/
structure DrawnBalls :=
  (fromA : Label)
  (fromB : Label)

/-- The sample space of all possible outcomes -/
def sampleSpace : List DrawnBalls := sorry

/-- Event A: sum of labels < 4 -/
def eventA (db : DrawnBalls) : Prop := sorry

/-- Event C: product of labels > 3 -/
def eventC (db : DrawnBalls) : Prop := sorry

/-- The probability of an event -/
def probability (event : DrawnBalls → Prop) : ℚ := sorry

theorem probability_of_event_A_is_half :
  probability eventA = 1 / 2 := sorry

theorem events_A_and_C_mutually_exclusive :
  ∀ db : DrawnBalls, ¬(eventA db ∧ eventC db) := sorry

end probability_of_event_A_is_half_events_A_and_C_mutually_exclusive_l686_68612
