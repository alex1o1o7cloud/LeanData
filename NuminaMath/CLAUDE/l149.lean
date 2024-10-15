import Mathlib

namespace NUMINAMATH_CALUDE_sum_inequality_l149_14965

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) : a + b + c ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l149_14965


namespace NUMINAMATH_CALUDE_vector_dot_product_properties_l149_14962

/-- Given two vectors in ℝ², prove dot product properties --/
theorem vector_dot_product_properties (a b : ℝ × ℝ) 
    (h1 : a = (1, 2)) 
    (h2 : b = (2, -3)) : 
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 9) ∧ 
  ((a.1 + b.1) * (a.1 - (1/9) * b.1) + (a.2 + b.2) * (a.2 - (1/9) * b.2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_properties_l149_14962


namespace NUMINAMATH_CALUDE_smallest_value_of_sum_of_cubes_l149_14933

theorem smallest_value_of_sum_of_cubes (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 2) 
  (h2 : Complex.abs (u^2 + v^2) = 8) : 
  Complex.abs (u^3 + v^3) = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_value_of_sum_of_cubes_l149_14933


namespace NUMINAMATH_CALUDE_no_intersection_implies_k_plus_minus_one_l149_14952

theorem no_intersection_implies_k_plus_minus_one (k : ℤ) :
  (∀ x y : ℝ, x^2 + y^2 = k^2 → x * y ≠ k) →
  k = 1 ∨ k = -1 := by
sorry

end NUMINAMATH_CALUDE_no_intersection_implies_k_plus_minus_one_l149_14952


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l149_14987

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Checks if an ellipse is tangent to both x and y axes -/
def isTangentToAxes (e : Ellipse) : Prop := sorry

/-- Calculates the length of the major axis of an ellipse -/
def majorAxisLength (e : Ellipse) : ℝ := sorry

/-- Main theorem: The length of the major axis of the given ellipse is 10 -/
theorem ellipse_major_axis_length :
  ∀ (e : Ellipse),
    e.focus1 = ⟨3, -5 + 2 * Real.sqrt 2⟩ ∧
    e.focus2 = ⟨3, -5 - 2 * Real.sqrt 2⟩ ∧
    isTangentToAxes e →
    majorAxisLength e = 10 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l149_14987


namespace NUMINAMATH_CALUDE_parabola_vertex_l149_14974

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = x^2 - 2x + 3 is (1, 2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, f x = (x - vertex.1)^2 + vertex.2) ∧ 
  (∀ x : ℝ, f x ≥ f vertex.1) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l149_14974


namespace NUMINAMATH_CALUDE_range_inequalities_l149_14902

theorem range_inequalities 
  (a b x y : ℝ) 
  (ha : 12 < a ∧ a < 60) 
  (hb : 15 < b ∧ b < 36) 
  (hxy1 : -1/2 < x - y ∧ x - y < 1/2) 
  (hxy2 : 0 < x + y ∧ x + y < 1) : 
  (-12 < 2*a - b ∧ 2*a - b < 105) ∧ 
  (1/3 < a/b ∧ a/b < 4) ∧ 
  (-1 < 3*x - y ∧ 3*x - y < 2) := by
sorry

end NUMINAMATH_CALUDE_range_inequalities_l149_14902


namespace NUMINAMATH_CALUDE_sequence_value_l149_14922

theorem sequence_value (a : ℕ → ℚ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = n / (n + 1)) :
  a 8 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_value_l149_14922


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_solution_l149_14905

-- Define a complex number to be purely imaginary if its real part is zero
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem complex_purely_imaginary_solution (z : ℂ) 
  (h1 : is_purely_imaginary z) 
  (h2 : is_purely_imaginary ((z - 3)^2 + 5*I)) : 
  z = 3*I ∨ z = -3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_solution_l149_14905


namespace NUMINAMATH_CALUDE_valid_selection_probability_l149_14944

/-- Represents a glove with a color and handedness -/
structure Glove :=
  (color : Nat)
  (isLeft : Bool)

/-- Represents a pair of gloves -/
structure GlovePair :=
  (left : Glove)
  (right : Glove)

/-- The set of all glove pairs in the cabinet -/
def glovePairs : Finset GlovePair := sorry

/-- The total number of ways to select two gloves -/
def totalSelections : Nat := sorry

/-- The number of valid selections (one left, one right, different pairs) -/
def validSelections : Nat := sorry

/-- The probability of a valid selection -/
def probabilityValidSelection : Rat := sorry

theorem valid_selection_probability :
  glovePairs.card = 3 →
  (∀ p : GlovePair, p ∈ glovePairs → p.left.color = p.right.color) →
  (∀ p q : GlovePair, p ∈ glovePairs → q ∈ glovePairs → p ≠ q → p.left.color ≠ q.left.color) →
  probabilityValidSelection = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_valid_selection_probability_l149_14944


namespace NUMINAMATH_CALUDE_product_increased_by_amount_l149_14991

theorem product_increased_by_amount (x y : ℝ) (h1 : x = 3) (h2 : 5 * x + y = 19) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_increased_by_amount_l149_14991


namespace NUMINAMATH_CALUDE_only_one_implies_negation_l149_14927

theorem only_one_implies_negation (p q : Prop) : 
  (∃! x : Fin 4, match x with
    | 0 => (p ∨ q) → ¬(p ∨ q)
    | 1 => (p ∧ ¬q) → ¬(p ∨ q)
    | 2 => (¬p ∧ q) → ¬(p ∨ q)
    | 3 => (¬p ∧ ¬q) → ¬(p ∨ q)
  ) := by sorry

end NUMINAMATH_CALUDE_only_one_implies_negation_l149_14927


namespace NUMINAMATH_CALUDE_zuminglish_word_count_mod_500_l149_14951

/-- Represents the alphabet of Zuminglish --/
inductive ZuminglishLetter
| M
| O
| P

/-- Represents whether a letter is a vowel or consonant --/
def isVowel (l : ZuminglishLetter) : Bool :=
  match l with
  | ZuminglishLetter.O => true
  | _ => false

/-- A Zuminglish word is a list of ZuminglishLetters --/
def ZuminglishWord := List ZuminglishLetter

/-- Check if a Zuminglish word is valid --/
def isValidWord (w : ZuminglishWord) : Bool :=
  sorry

/-- Count the number of valid 10-letter Zuminglish words --/
def countValidWords : Nat :=
  sorry

/-- The main theorem to prove --/
theorem zuminglish_word_count_mod_500 :
  countValidWords % 500 = 160 :=
sorry

end NUMINAMATH_CALUDE_zuminglish_word_count_mod_500_l149_14951


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_line_l149_14950

/-- The shortest distance between a point on the parabola y = x^2 - 4x + 11 
    and a point on the line y = 2x - 6 is 8/√5. -/
theorem shortest_distance_parabola_line : 
  let parabola := {P : ℝ × ℝ | P.2 = P.1^2 - 4*P.1 + 11}
  let line := {Q : ℝ × ℝ | Q.2 = 2*Q.1 - 6}
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ parabola ∧ Q ∈ line ∧
    ∀ (P' : ℝ × ℝ) (Q' : ℝ × ℝ), P' ∈ parabola → Q' ∈ line →
      Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ 8 / Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_line_l149_14950


namespace NUMINAMATH_CALUDE_quiz_score_proof_l149_14908

theorem quiz_score_proof (score1 score2 score3 : ℕ) : 
  score2 = 90 → score3 = 92 → (score1 + score2 + score3) / 3 = 91 → score1 = 91 := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_proof_l149_14908


namespace NUMINAMATH_CALUDE_subset_implies_m_leq_two_l149_14953

def A : Set ℝ := {x | x < 2}
def B (m : ℝ) : Set ℝ := {x | x < m}

theorem subset_implies_m_leq_two (m : ℝ) : B m ⊆ A → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_leq_two_l149_14953


namespace NUMINAMATH_CALUDE_invoice_error_correction_l149_14966

/-- Two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The proposition to be proved -/
theorem invoice_error_correction (x y : ℕ) 
  (hx : TwoDigitNumber x) (hy : TwoDigitNumber y)
  (h_diff : 100 * x + y - (100 * y + x) = 3654) :
  x = 63 ∧ y = 26 := by
  sorry

end NUMINAMATH_CALUDE_invoice_error_correction_l149_14966


namespace NUMINAMATH_CALUDE_fraction_equality_l149_14984

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 4) :
  (4 * a + 3 * b) / (4 * a - 3 * b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l149_14984


namespace NUMINAMATH_CALUDE_sum_positive_implies_both_positive_is_false_l149_14992

theorem sum_positive_implies_both_positive_is_false : 
  ¬(∀ a b : ℝ, a + b > 0 → a > 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_both_positive_is_false_l149_14992


namespace NUMINAMATH_CALUDE_min_positive_temperatures_l149_14912

theorem min_positive_temperatures (n : ℕ) (pos_products neg_products : ℕ) : 
  n = 12 → 
  pos_products = 78 → 
  neg_products = 54 → 
  pos_products + neg_products = n * (n - 1) →
  ∃ y : ℕ, y ≥ 3 ∧ y * (y - 1) + (n - y) * (n - 1 - y) = pos_products ∧
  ∀ z : ℕ, z < 3 → z * (z - 1) + (n - z) * (n - 1 - z) ≠ pos_products :=
by sorry

end NUMINAMATH_CALUDE_min_positive_temperatures_l149_14912


namespace NUMINAMATH_CALUDE_green_tile_probability_l149_14931

theorem green_tile_probability :
  let total_tiles := 100
  let is_green (n : ℕ) := n % 5 = 3
  let green_tiles := Finset.filter is_green (Finset.range total_tiles)
  (green_tiles.card : ℚ) / total_tiles = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_green_tile_probability_l149_14931


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l149_14934

theorem subtraction_of_decimals : 3.57 - 1.45 = 2.12 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l149_14934


namespace NUMINAMATH_CALUDE_red_faces_cube_l149_14995

theorem red_faces_cube (n : ℕ) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/4 ↔ n = 4 := by sorry

end NUMINAMATH_CALUDE_red_faces_cube_l149_14995


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l149_14997

theorem quadratic_roots_properties (m : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 - 2*(m+1)*x + m^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  (m ≥ 1 ∧ ∃ m', m' ≥ 1 ∧ (x₁ - 1)*(x₂ - 1) = m' + 6 ∧ m' = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l149_14997


namespace NUMINAMATH_CALUDE_exists_max_a_l149_14907

def is_valid_number (a d e : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 8 ∧
  (500000 + 100000 * a + 1000 * d + 500 + 20 + 4 + e) % 24 = 0

theorem exists_max_a : ∃ (d e : ℕ), is_valid_number 9 d e :=
sorry

end NUMINAMATH_CALUDE_exists_max_a_l149_14907


namespace NUMINAMATH_CALUDE_complex_number_location_l149_14978

theorem complex_number_location : 
  let z : ℂ := 1 - (1 / Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l149_14978


namespace NUMINAMATH_CALUDE_parking_ticket_ratio_l149_14993

/-- Represents the number of tickets for each person -/
structure Tickets where
  parking : ℕ
  speeding : ℕ

/-- The problem setup -/
def ticketProblem (mark sarah : Tickets) : Prop :=
  mark.speeding = sarah.speeding ∧
  sarah.speeding = 6 ∧
  mark.parking = 8 ∧
  mark.parking + mark.speeding + sarah.parking + sarah.speeding = 24

/-- The theorem to prove -/
theorem parking_ticket_ratio (mark sarah : Tickets) 
  (h : ticketProblem mark sarah) : 
  mark.parking * 1 = sarah.parking * 2 := by
  sorry


end NUMINAMATH_CALUDE_parking_ticket_ratio_l149_14993


namespace NUMINAMATH_CALUDE_geometric_series_proof_l149_14909

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_proof :
  let a : ℚ := 1/2
  let r : ℚ := -1/2
  let n : ℕ := 6
  geometric_series_sum a r n = 21/64 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_series_proof_l149_14909


namespace NUMINAMATH_CALUDE_max_pieces_theorem_l149_14935

/-- Represents the size of the cake in inches -/
def cake_size : ℕ := 100

/-- Represents the size of each piece in inches -/
def piece_size : ℕ := 4

/-- Predicate to check if a number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Theorem stating the maximum number of pieces that can be cut from the cake -/
theorem max_pieces_theorem :
  (is_even cake_size) →
  (is_even piece_size) →
  (cake_size % piece_size = 0) →
  (cake_size / piece_size) * (cake_size / piece_size) = 625 := by
  sorry

#check max_pieces_theorem

end NUMINAMATH_CALUDE_max_pieces_theorem_l149_14935


namespace NUMINAMATH_CALUDE_lauren_reaches_andrea_l149_14976

/-- The initial distance between Andrea and Lauren in kilometers -/
def initial_distance : ℝ := 30

/-- The rate at which the distance between Andrea and Lauren decreases in km/min -/
def distance_decrease_rate : ℝ := 2

/-- The duration of initial biking in minutes -/
def initial_biking_time : ℝ := 10

/-- The duration of the stop in minutes -/
def stop_time : ℝ := 5

/-- Andrea's speed in km/h -/
def andrea_speed : ℝ := 40

/-- Lauren's speed in km/h -/
def lauren_speed : ℝ := 80

/-- The total time it takes for Lauren to reach Andrea -/
def total_time : ℝ := 22.5

theorem lauren_reaches_andrea :
  let distance_covered := distance_decrease_rate * initial_biking_time
  let remaining_distance := initial_distance - distance_covered
  let lauren_final_time := remaining_distance / (lauren_speed / 60)
  total_time = initial_biking_time + stop_time + lauren_final_time :=
by
  sorry

#check lauren_reaches_andrea

end NUMINAMATH_CALUDE_lauren_reaches_andrea_l149_14976


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l149_14961

theorem min_value_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 5) :
  (1 / x + 4 / y + 9 / z) ≥ 36 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l149_14961


namespace NUMINAMATH_CALUDE_polygon_and_calendar_problem_l149_14906

theorem polygon_and_calendar_problem :
  ∀ (n k : ℕ),
  -- Regular polygon with interior angles of 160°
  (180 - 160 : ℝ) * n = 360 →
  -- The n-th day of May is Friday
  n % 7 = 5 →
  -- The k-th day of May is Tuesday
  k % 7 = 2 →
  -- 20 < k < 26
  20 < k ∧ k < 26 →
  -- Prove n = 18 and k = 22
  n = 18 ∧ k = 22 :=
by sorry

end NUMINAMATH_CALUDE_polygon_and_calendar_problem_l149_14906


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_sum_of_digits_l149_14994

/-- Function to create a number with n ones -/
def ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Function to calculate the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Theorem: For all natural numbers n, the number formed by 3^n ones is divisible by the sum of its digits -/
theorem infinitely_many_divisible_by_sum_of_digits (n : ℕ) :
  ∃ (k : ℕ), k > 0 ∧ (ones (3^n) % sumOfDigits (ones (3^n)) = 0) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_sum_of_digits_l149_14994


namespace NUMINAMATH_CALUDE_horner_method_correct_l149_14937

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 1 + x + 2x^2 + 3x^3 + 4x^4 + 5x^5 -/
def f (x : ℝ) : ℝ :=
  1 + x + 2*x^2 + 3*x^3 + 4*x^4 + 5*x^5

theorem horner_method_correct :
  horner [5, 4, 3, 2, 1, 1] (-1) = f (-1) ∧ f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_correct_l149_14937


namespace NUMINAMATH_CALUDE_locus_equation_l149_14929

/-- Parabola type representing y^2 = 4px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (par : Parabola) where
  y : ℝ
  eq : y^2 = 4 * par.p * (y^2 / (4 * par.p))

/-- The locus of point M given two points on a parabola -/
def locusM (par : Parabola) (A B : ParabolaPoint par) (M : ℝ × ℝ) : Prop :=
  let OA := (A.y^2 / (4 * par.p), A.y)
  let OB := (B.y^2 / (4 * par.p), B.y)
  let (x, y) := M
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) ∧  -- OA ⊥ OB
  (x * (B.y^2 - A.y^2) / (4 * par.p) + y * (B.y - A.y) = 0) ∧  -- OM ⊥ AB
  (x - A.y^2 / (4 * par.p)) * (B.y - A.y) = 
    ((B.y^2 - A.y^2) / (4 * par.p)) * (y - A.y)  -- M is on line AB

theorem locus_equation (par : Parabola) (A B : ParabolaPoint par) (M : ℝ × ℝ) :
  locusM par A B M → M.1^2 + M.2^2 - 4 * par.p * M.1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_equation_l149_14929


namespace NUMINAMATH_CALUDE_inscribed_circle_tangent_sum_l149_14986

/-- A point on the inscribed circle of a square -/
structure InscribedCirclePoint (α β : ℝ) where
  -- P is on the inscribed circle of square ABCD
  on_inscribed_circle : True
  -- Angle APC = α
  angle_apc : True
  -- Angle BPD = β
  angle_bpd : True

/-- The sum of squared tangents of angles α and β is 8 -/
theorem inscribed_circle_tangent_sum (α β : ℝ) (p : InscribedCirclePoint α β) : 
  Real.tan α ^ 2 + Real.tan β ^ 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_tangent_sum_l149_14986


namespace NUMINAMATH_CALUDE_dog_catches_fox_dog_catches_fox_specific_l149_14955

/-- The distance at which a dog catches a fox given initial conditions -/
theorem dog_catches_fox (initial_distance : ℝ) (dog_leap : ℝ) (fox_leap : ℝ) 
  (dog_leaps_per_unit : ℕ) (fox_leaps_per_unit : ℕ) : ℝ :=
  let dog_distance_per_unit := dog_leap * dog_leaps_per_unit
  let fox_distance_per_unit := fox_leap * fox_leaps_per_unit
  let relative_distance_per_unit := dog_distance_per_unit - fox_distance_per_unit
  let time_units_to_catch := initial_distance / relative_distance_per_unit
  time_units_to_catch * dog_distance_per_unit

/-- The specific case of the dog catching the fox problem -/
theorem dog_catches_fox_specific : 
  dog_catches_fox 30 2 1 2 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_dog_catches_fox_dog_catches_fox_specific_l149_14955


namespace NUMINAMATH_CALUDE_smallest_nonprime_no_small_factors_range_l149_14932

-- Define the property of having no prime factors less than 20
def no_small_prime_factors (n : ℕ) : Prop :=
  ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

-- Define the property of being the smallest nonprime with no small prime factors
def smallest_nonprime_no_small_factors (n : ℕ) : Prop :=
  n > 1 ∧ ¬n.Prime ∧ no_small_prime_factors n ∧
  ∀ m, m > 1 → ¬m.Prime → no_small_prime_factors m → n ≤ m

-- State the theorem
theorem smallest_nonprime_no_small_factors_range :
  ∃ n, smallest_nonprime_no_small_factors n ∧ 500 < n ∧ n ≤ 550 := by
  sorry

end NUMINAMATH_CALUDE_smallest_nonprime_no_small_factors_range_l149_14932


namespace NUMINAMATH_CALUDE_symmetry_about_y_axis_l149_14917

/-- Given two real numbers a and b such that log(a) + log(b) = 0, a ≠ 1, and b ≠ 1,
    prove that the functions f(x) = ax and g(x) = bx are symmetric about the y-axis. -/
theorem symmetry_about_y_axis 
  (a b : ℝ) 
  (h1 : Real.log a + Real.log b = 0) 
  (h2 : a ≠ 1) 
  (h3 : b ≠ 1) : 
  ∀ x : ℝ, ∃ y : ℝ, a * x = b * (-y) ∧ a * (-x) = b * y :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_y_axis_l149_14917


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l149_14957

theorem exponential_function_fixed_point (a b : ℝ) (ha : a > 0) :
  (∀ x, (a^(x - b) + 1 = 2) ↔ (x = 1)) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l149_14957


namespace NUMINAMATH_CALUDE_cannot_determine_books_left_l149_14939

def initial_pens : ℕ := 42
def initial_books : ℕ := 143
def pens_sold : ℕ := 23
def pens_left : ℕ := 19

theorem cannot_determine_books_left : 
  ∀ (books_left : ℕ), 
  initial_pens = pens_sold + pens_left →
  ¬(∀ (books_sold : ℕ), initial_books = books_sold + books_left) :=
by
  sorry

end NUMINAMATH_CALUDE_cannot_determine_books_left_l149_14939


namespace NUMINAMATH_CALUDE_expected_worth_unfair_coin_l149_14923

/-- An unfair coin with given probabilities and payoffs -/
structure UnfairCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  payoff_heads : ℝ
  payoff_tails : ℝ
  prob_sum : prob_heads + prob_tails = 1

/-- The expected worth of a coin flip -/
def expected_worth (c : UnfairCoin) : ℝ :=
  c.prob_heads * c.payoff_heads + c.prob_tails * c.payoff_tails

/-- Theorem stating the expected worth of the specific unfair coin -/
theorem expected_worth_unfair_coin :
  ∃ c : UnfairCoin, c.prob_heads = 3/4 ∧ c.prob_tails = 1/4 ∧
  c.payoff_heads = 3 ∧ c.payoff_tails = -8 ∧ expected_worth c = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_unfair_coin_l149_14923


namespace NUMINAMATH_CALUDE_product_equals_eighteen_l149_14911

theorem product_equals_eighteen : 12 * 0.5 * 3 * 0.2 * 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eighteen_l149_14911


namespace NUMINAMATH_CALUDE_fifth_boy_payment_is_35_l149_14904

/-- The total cost of the video game system -/
def total_cost : ℚ := 120

/-- The amount paid by the fourth boy -/
def fourth_boy_payment : ℚ := 20

/-- The payment fractions for the first three boys -/
def first_boy_fraction : ℚ := 1/3
def second_boy_fraction : ℚ := 1/4
def third_boy_fraction : ℚ := 1/5

/-- The amounts paid by each boy -/
noncomputable def first_boy_payment (second third fourth fifth : ℚ) : ℚ :=
  first_boy_fraction * (second + third + fourth + fifth)

noncomputable def second_boy_payment (first third fourth fifth : ℚ) : ℚ :=
  second_boy_fraction * (first + third + fourth + fifth)

noncomputable def third_boy_payment (first second fourth fifth : ℚ) : ℚ :=
  third_boy_fraction * (first + second + fourth + fifth)

/-- The theorem stating that the fifth boy paid $35 -/
theorem fifth_boy_payment_is_35 :
  ∃ (first second third fifth : ℚ),
    first = first_boy_payment second third fourth_boy_payment fifth ∧
    second = second_boy_payment first third fourth_boy_payment fifth ∧
    third = third_boy_payment first second fourth_boy_payment fifth ∧
    first + second + third + fourth_boy_payment + fifth = total_cost ∧
    fifth = 35 := by
  sorry

end NUMINAMATH_CALUDE_fifth_boy_payment_is_35_l149_14904


namespace NUMINAMATH_CALUDE_equation_represents_three_non_concurrent_lines_l149_14990

/-- The equation represents three lines that do not all pass through a common point -/
theorem equation_represents_three_non_concurrent_lines :
  ∃ (l₁ l₂ l₃ : ℝ → ℝ → Prop),
    (∀ x y, (x^2 - 3*y)*(x - y + 1) = (y^2 - 3*x)*(x - y + 1) ↔ l₁ x y ∨ l₂ x y ∨ l₃ x y) ∧
    (∃ x₁ y₁, l₁ x₁ y₁ ∧ l₂ x₁ y₁ ∧ ¬l₃ x₁ y₁) ∧
    (∃ x₂ y₂, l₁ x₂ y₂ ∧ ¬l₂ x₂ y₂ ∧ l₃ x₂ y₂) ∧
    (∃ x₃ y₃, ¬l₁ x₃ y₃ ∧ l₂ x₃ y₃ ∧ l₃ x₃ y₃) ∧
    (∀ x y, ¬(l₁ x y ∧ l₂ x y ∧ l₃ x y)) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_represents_three_non_concurrent_lines_l149_14990


namespace NUMINAMATH_CALUDE_foreign_trade_income_equation_l149_14973

/-- The foreign trade income equation over two years with a constant growth rate -/
theorem foreign_trade_income_equation
  (m : ℝ) -- foreign trade income in 2001 (billion yuan)
  (x : ℝ) -- annual growth rate
  (n : ℝ) -- foreign trade income in 2003 (billion yuan)
  : m * (1 + x)^2 = n :=
by sorry

end NUMINAMATH_CALUDE_foreign_trade_income_equation_l149_14973


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l149_14928

theorem parabola_focus_distance (p : ℝ) (h1 : p > 0) :
  ∃ (x y : ℝ),
    y^2 = 2*p*x ∧
    x + p/2 = 2 →
    Real.sqrt (x - p/2)^2 + y^2 = Real.sqrt (2*p*(2 - p/2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l149_14928


namespace NUMINAMATH_CALUDE_machine_a_production_time_l149_14914

/-- The time (in minutes) it takes for Machine A to produce one item -/
def t : ℝ := sorry

/-- The time (in minutes) it takes for Machine B to produce one item -/
def machine_b_time : ℝ := 5

/-- The duration of the production period in minutes -/
def production_period : ℝ := 1440

/-- The ratio of items produced by Machine A compared to Machine B -/
def production_ratio : ℝ := 1.25

theorem machine_a_production_time : 
  (production_period / t = production_ratio * (production_period / machine_b_time)) → t = 4 := by
  sorry

end NUMINAMATH_CALUDE_machine_a_production_time_l149_14914


namespace NUMINAMATH_CALUDE_total_spent_is_40_l149_14946

def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def num_ingredients : ℕ := 5
def apron_cost : ℕ := recipe_book_cost + 1

def total_cost : ℕ := recipe_book_cost + baking_dish_cost + (ingredient_cost * num_ingredients) + apron_cost

theorem total_spent_is_40 : total_cost = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_40_l149_14946


namespace NUMINAMATH_CALUDE_smallest_triangle_leg_l149_14948

/-- Represents a 45-45-90 triangle -/
structure Triangle45 where
  hypotenuse : ℝ
  leg : ℝ
  hyp_leg_relation : leg = hypotenuse / Real.sqrt 2

/-- A sequence of four 45-45-90 triangles where the hypotenuse of one is the leg of the next -/
def TriangleSequence (t1 t2 t3 t4 : Triangle45) : Prop :=
  t1.leg = t2.hypotenuse ∧ t2.leg = t3.hypotenuse ∧ t3.leg = t4.hypotenuse

theorem smallest_triangle_leg 
  (t1 t2 t3 t4 : Triangle45) 
  (seq : TriangleSequence t1 t2 t3 t4) 
  (largest_hyp : t1.hypotenuse = 16) : 
  t4.leg = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_leg_l149_14948


namespace NUMINAMATH_CALUDE_possible_tile_counts_l149_14941

/-- Represents the dimensions of a rectangular floor in terms of tiles -/
structure FloorDimensions where
  width : ℕ
  length : ℕ

/-- Calculates the number of red tiles on the floor -/
def redTiles (d : FloorDimensions) : ℕ := 2 * d.width + 2 * d.length - 4

/-- Calculates the number of white tiles on the floor -/
def whiteTiles (d : FloorDimensions) : ℕ := d.width * d.length - redTiles d

/-- Checks if the number of red and white tiles are equal -/
def equalRedWhite (d : FloorDimensions) : Prop := redTiles d = whiteTiles d

/-- The theorem stating the possible total number of tiles -/
theorem possible_tile_counts : 
  ∀ d : FloorDimensions, 
    equalRedWhite d → 
    d.width * d.length = 48 ∨ d.width * d.length = 60 := by
  sorry

end NUMINAMATH_CALUDE_possible_tile_counts_l149_14941


namespace NUMINAMATH_CALUDE_tangent_through_origin_l149_14999

/-- The curve y = x^α + 1 has a tangent line at (1, 2) that passes through the origin if and only if α = 2 -/
theorem tangent_through_origin (α : ℝ) : 
  (∃ (m : ℝ), (∀ x : ℝ, x^α + 1 = m * (x - 1) + 2) ∧ m * (-1) + 2 = 0) ↔ α = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_through_origin_l149_14999


namespace NUMINAMATH_CALUDE_greatest_x_satisfying_equation_l149_14960

theorem greatest_x_satisfying_equation : 
  ∃ (x : ℝ), x = -3 ∧ 
  (∀ y : ℝ, y ≠ 6 → y ≠ -4 → (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≤ x) ∧
  (x^2 - x - 30) / (x - 6) = 2 / (x + 4) ∧
  x ≠ 6 ∧ x ≠ -4 := by
sorry

end NUMINAMATH_CALUDE_greatest_x_satisfying_equation_l149_14960


namespace NUMINAMATH_CALUDE_similar_triangles_with_two_equal_sides_l149_14930

theorem similar_triangles_with_two_equal_sides (a b c d e f : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  (a = 80 ∧ b = 100) →
  (d = 80 ∧ e = 100) →
  a / d = b / e →
  a / d = c / f →
  b / e = c / f →
  ((c = 64 ∧ f = 125) ∨ (c = 125 ∧ f = 64)) :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_with_two_equal_sides_l149_14930


namespace NUMINAMATH_CALUDE_same_solution_implies_a_b_values_l149_14915

theorem same_solution_implies_a_b_values :
  ∀ (a b x y : ℚ),
  (3 * x - y = 7 ∧ a * x + y = b) ∧
  (x + b * y = a ∧ 2 * x + y = 8) →
  a = -7/5 ∧ b = -11/5 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_b_values_l149_14915


namespace NUMINAMATH_CALUDE_valid_paintings_count_l149_14947

/-- Represents a 3x3 grid of squares that can be painted green or red -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Bool :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Checks if a grid painting is valid (no green square adjacent to a red square) -/
def valid_painting (g : Grid) : Bool :=
  ∀ p1 p2 : Fin 3 × Fin 3, adjacent p1 p2 → (g p1.1 p1.2 = g p2.1 p2.2)

/-- Counts the number of valid grid paintings -/
def count_valid_paintings : Nat :=
  (List.filter valid_painting (List.map (λf : Fin 9 → Bool => λi j => f (3 * i + j)) 
    (List.map (λn : Fin 512 => λi => n.val.testBit i) (List.range 512)))).length

/-- The main theorem stating that the number of valid paintings is 10 -/
theorem valid_paintings_count : count_valid_paintings = 10 := by
  sorry

end NUMINAMATH_CALUDE_valid_paintings_count_l149_14947


namespace NUMINAMATH_CALUDE_employee_not_working_first_day_l149_14903

def total_employees : ℕ := 6
def days : ℕ := 3
def employees_per_day : ℕ := 2

def schedule_probability (n m : ℕ) : ℚ := (n.choose m : ℚ) / (total_employees.choose employees_per_day : ℚ)

theorem employee_not_working_first_day :
  schedule_probability (total_employees - 1) employees_per_day = 2/3 :=
sorry

end NUMINAMATH_CALUDE_employee_not_working_first_day_l149_14903


namespace NUMINAMATH_CALUDE_music_shop_total_cost_l149_14958

/-- Calculates the total cost of CDs purchased from a music shop --/
theorem music_shop_total_cost 
  (life_journey_price : ℝ) 
  (life_journey_discount : ℝ) 
  (day_life_price : ℝ) 
  (rescind_price : ℝ) 
  (life_journey_quantity : ℕ) 
  (day_life_quantity : ℕ) 
  (rescind_quantity : ℕ) : 
  life_journey_price = 100 →
  life_journey_discount = 0.2 →
  day_life_price = 50 →
  rescind_price = 85 →
  life_journey_quantity = 3 →
  day_life_quantity = 4 →
  rescind_quantity = 2 →
  (life_journey_quantity * (life_journey_price * (1 - life_journey_discount))) +
  ((day_life_quantity / 2) * day_life_price) +
  (rescind_quantity * rescind_price) = 510 := by
sorry

end NUMINAMATH_CALUDE_music_shop_total_cost_l149_14958


namespace NUMINAMATH_CALUDE_sqrt_three_div_sqrt_one_third_eq_three_l149_14971

theorem sqrt_three_div_sqrt_one_third_eq_three : 
  Real.sqrt 3 / Real.sqrt (1/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_div_sqrt_one_third_eq_three_l149_14971


namespace NUMINAMATH_CALUDE_kevin_ran_17_miles_l149_14954

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Kevin's total running distance -/
def kevin_total_distance : ℝ :=
  let segment1 := distance 10 0.5
  let segment2 := distance 20 0.5
  let segment3 := distance 8 0.25
  segment1 + segment2 + segment3

/-- Theorem stating that Kevin's total running distance is 17 miles -/
theorem kevin_ran_17_miles : kevin_total_distance = 17 := by
  sorry

end NUMINAMATH_CALUDE_kevin_ran_17_miles_l149_14954


namespace NUMINAMATH_CALUDE_zoo_animals_l149_14970

/-- The number of ostriches in the zoo -/
def num_ostriches : ℕ := 15

/-- The number of sika deer in the zoo -/
def num_deer : ℕ := 23

/-- The number of legs an ostrich has -/
def ostrich_legs : ℕ := 2

/-- The number of legs a sika deer has -/
def deer_legs : ℕ := 4

/-- The total number of legs of all animals -/
def total_legs : ℕ := 122

/-- The total number of legs if the numbers of ostriches and deer were swapped -/
def swapped_legs : ℕ := 106

theorem zoo_animals :
  num_ostriches * ostrich_legs + num_deer * deer_legs = total_legs ∧
  num_deer * ostrich_legs + num_ostriches * deer_legs = swapped_legs :=
by sorry

end NUMINAMATH_CALUDE_zoo_animals_l149_14970


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l149_14901

theorem magnitude_of_complex_fraction (i : ℂ) (h : i ^ 2 = -1) :
  Complex.abs (i / (2 - i)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l149_14901


namespace NUMINAMATH_CALUDE_gym_class_laps_l149_14900

/-- Given a total distance to run, track length, and number of laps already run by two people,
    calculate the number of additional laps needed to reach the total distance. -/
def additional_laps_needed (total_distance : ℕ) (track_length : ℕ) (laps_run_per_person : ℕ) : ℕ :=
  let total_laps_run := 2 * laps_run_per_person
  let distance_run := total_laps_run * track_length
  let remaining_distance := total_distance - distance_run
  remaining_distance / track_length

/-- Prove that for the given conditions, the number of additional laps needed is 4. -/
theorem gym_class_laps : additional_laps_needed 2400 150 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gym_class_laps_l149_14900


namespace NUMINAMATH_CALUDE_cross_number_puzzle_l149_14936

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_power_of_3 (n : ℕ) : Prop := ∃ m : ℕ, n = 3^m

def is_power_of_7 (n : ℕ) : Prop := ∃ m : ℕ, n = 7^m

def digit_in_number (d : ℕ) (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 10 + d + b * 100 ∧ d < 10

theorem cross_number_puzzle :
  ∃! d : ℕ, 
    (∃ n : ℕ, is_three_digit n ∧ is_power_of_3 n ∧ digit_in_number d n) ∧
    (∃ m : ℕ, is_three_digit m ∧ is_power_of_7 m ∧ digit_in_number d m) ∧
    d = 4 :=
sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_l149_14936


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l149_14916

theorem arithmetic_mean_problem (p q r : ℝ) : 
  (p + q) / 2 = 10 →
  (q + r) / 2 = 27 →
  r - p = 34 →
  (q + r) / 2 = 27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l149_14916


namespace NUMINAMATH_CALUDE_square_area_comparison_l149_14969

theorem square_area_comparison (a b : ℝ) (h : b = 4 * a) :
  b ^ 2 = 16 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_comparison_l149_14969


namespace NUMINAMATH_CALUDE_square_root_fraction_equality_l149_14945

theorem square_root_fraction_equality : 
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = 17 / Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_equality_l149_14945


namespace NUMINAMATH_CALUDE_dime_probability_l149_14998

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℚ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℚ
  | _ => 1250

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℚ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℚ := coinCount Coin.Quarter + coinCount Coin.Dime + coinCount Coin.Penny

/-- The probability of selecting a dime from the jar -/
def probDime : ℚ := coinCount Coin.Dime / totalCoins

theorem dime_probability : probDime = 5 / 57 := by
  sorry

end NUMINAMATH_CALUDE_dime_probability_l149_14998


namespace NUMINAMATH_CALUDE_friends_pooling_money_l149_14919

-- Define the friends
inductive Friend
| Emma
| Daya
| Jeff
| Brenda

-- Define a function to get the amount of money each friend has
def money (f : Friend) : ℚ :=
  match f with
  | Friend.Emma => 8
  | Friend.Daya => 8 * (1 + 1/4)
  | Friend.Jeff => (2/5) * (8 * (1 + 1/4))
  | Friend.Brenda => (2/5) * (8 * (1 + 1/4)) + 4

-- Theorem stating that there are 4 friends pooling money for pizza
theorem friends_pooling_money :
  (∃ (s : Finset Friend), s.card = 4 ∧ 
    (∀ f : Friend, f ∈ s) ∧
    (money Friend.Emma = 8) ∧
    (money Friend.Daya = money Friend.Emma * (1 + 1/4)) ∧
    (money Friend.Jeff = (2/5) * money Friend.Daya) ∧
    (money Friend.Brenda = money Friend.Jeff + 4) ∧
    (money Friend.Brenda = 8)) :=
by sorry

end NUMINAMATH_CALUDE_friends_pooling_money_l149_14919


namespace NUMINAMATH_CALUDE_even_function_sum_l149_14959

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_sum (f : ℝ → ℝ) (h_even : is_even_function f) (h_f4 : f 4 = 5) :
  f 4 + f (-4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l149_14959


namespace NUMINAMATH_CALUDE_milk_production_l149_14964

/-- Milk production calculation -/
theorem milk_production
  (a b c d e f : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
  (h_efficiency : 0 < f ∧ f ≤ 100)
  (h_initial : b = (a * c) * (b / (a * c)))  -- Initial production rate
  : (d * e) * ((b / (a * c)) * (f / 100)) = b * d * e * f / (100 * a * c) :=
by sorry

#check milk_production

end NUMINAMATH_CALUDE_milk_production_l149_14964


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l149_14972

/-- Two points are symmetric about the y-axis if their y-coordinates are equal and their x-coordinates are opposite -/
def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = y2 ∧ x1 = -x2

/-- The problem statement -/
theorem symmetric_points_difference (m n : ℝ) :
  symmetric_about_y_axis 3 m n 4 → m - n = 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l149_14972


namespace NUMINAMATH_CALUDE_total_students_l149_14924

/-- Represents the number of students in each grade --/
def Students := Fin 8 → ℕ

/-- The total number of students in grades I-IV is 130 --/
def sum_I_to_IV (s : Students) : Prop :=
  s 0 + s 1 + s 2 + s 3 = 130

/-- Grade V has 7 more students than grade II --/
def grade_V_condition (s : Students) : Prop :=
  s 4 = s 1 + 7

/-- Grade VI has 5 fewer students than grade I --/
def grade_VI_condition (s : Students) : Prop :=
  s 5 = s 0 - 5

/-- Grade VII has 10 more students than grade IV --/
def grade_VII_condition (s : Students) : Prop :=
  s 6 = s 3 + 10

/-- Grade VIII has 4 fewer students than grade I --/
def grade_VIII_condition (s : Students) : Prop :=
  s 7 = s 0 - 4

/-- The theorem stating that the total number of students is 268 --/
theorem total_students (s : Students)
  (h1 : sum_I_to_IV s)
  (h2 : grade_V_condition s)
  (h3 : grade_VI_condition s)
  (h4 : grade_VII_condition s)
  (h5 : grade_VIII_condition s) :
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 + s 7 = 268 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l149_14924


namespace NUMINAMATH_CALUDE_sum_of_variables_l149_14979

theorem sum_of_variables (a b c d e : ℝ) 
  (eq1 : 3*a + 2*b + 4*d = 10)
  (eq2 : 6*a + 5*b + 4*c + 3*d + 2*e = 8)
  (eq3 : a + b + 2*c + 5*e = 3)
  (eq4 : 2*c + 3*d + 3*e = 4)
  (eq5 : a + 2*b + 3*c + d = 7) :
  a + b + c + d + e = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_variables_l149_14979


namespace NUMINAMATH_CALUDE_bus_cyclist_speed_problem_l149_14981

/-- Proves that given the problem conditions, the speeds of the bus and cyclist are 35 km/h and 15 km/h respectively. -/
theorem bus_cyclist_speed_problem (distance : ℝ) (first_meeting_time : ℝ) (bus_stop_time : ℝ) (overtake_time : ℝ)
  (h1 : distance = 70)
  (h2 : first_meeting_time = 7/5)
  (h3 : bus_stop_time = 1/3)
  (h4 : overtake_time = 161/60) :
  ∃ (bus_speed cyclist_speed : ℝ),
    bus_speed = 35 ∧
    cyclist_speed = 15 ∧
    first_meeting_time * (bus_speed + cyclist_speed) = distance ∧
    (first_meeting_time + overtake_time - bus_stop_time) * bus_speed - (first_meeting_time + overtake_time) * cyclist_speed = distance :=
by sorry

end NUMINAMATH_CALUDE_bus_cyclist_speed_problem_l149_14981


namespace NUMINAMATH_CALUDE_paths_from_A_to_B_is_16_l149_14918

/-- Represents the number of arrows of each color in the hexagonal lattice --/
structure ArrowCounts where
  red : Nat
  blue : Nat
  green : Nat
  purple : Nat
  orange : Nat

/-- Represents the connection rules between arrows of different colors --/
structure ConnectionRules where
  redToBlue : Nat
  blueToGreen : Nat
  greenToPurple : Nat
  purpleToOrange : Nat
  orangeToB : Nat

/-- Calculates the number of paths from A to B in the hexagonal lattice --/
def pathsFromAToB (counts : ArrowCounts) (rules : ConnectionRules) : Nat :=
  counts.red * rules.redToBlue * counts.blue * rules.blueToGreen * counts.green *
  rules.greenToPurple * counts.purple * rules.purpleToOrange * counts.orange * rules.orangeToB

/-- Theorem stating that the number of paths from A to B is 16 --/
theorem paths_from_A_to_B_is_16 (counts : ArrowCounts) (rules : ConnectionRules) :
  counts.red = 2 ∧ counts.blue = 2 ∧ counts.green = 4 ∧ counts.purple = 4 ∧ counts.orange = 4 ∧
  rules.redToBlue = 2 ∧ rules.blueToGreen = 3 ∧ rules.greenToPurple = 2 ∧
  rules.purpleToOrange = 1 ∧ rules.orangeToB = 1 →
  pathsFromAToB counts rules = 16 := by
  sorry

#check paths_from_A_to_B_is_16

end NUMINAMATH_CALUDE_paths_from_A_to_B_is_16_l149_14918


namespace NUMINAMATH_CALUDE_q_div_p_eq_fifty_l149_14926

/-- The number of cards in the box -/
def total_cards : ℕ := 30

/-- The number of different numbers on the cards -/
def num_types : ℕ := 6

/-- The number of cards for each number -/
def cards_per_num : ℕ := 5

/-- The number of cards drawn -/
def drawn_cards : ℕ := 4

/-- The probability of drawing four cards with the same number -/
def p : ℚ := (num_types * (cards_per_num.choose drawn_cards)) / (total_cards.choose drawn_cards)

/-- The probability of drawing two pairs of cards with different numbers -/
def q : ℚ := (num_types.choose 2 * (cards_per_num.choose 2)^2) / (total_cards.choose drawn_cards)

/-- The theorem stating that the ratio of q to p is 50 -/
theorem q_div_p_eq_fifty : q / p = 50 := by sorry

end NUMINAMATH_CALUDE_q_div_p_eq_fifty_l149_14926


namespace NUMINAMATH_CALUDE_lucy_mother_age_relation_l149_14913

/-- Lucy's age in 2010 -/
def lucy_age_2010 : ℕ := 10

/-- Lucy's mother's age in 2010 -/
def mother_age_2010 : ℕ := 5 * lucy_age_2010

/-- The year when Lucy's mother's age will be twice Lucy's age -/
def target_year : ℕ := 2040

/-- The number of years from 2010 to the target year -/
def years_passed : ℕ := target_year - 2010

theorem lucy_mother_age_relation :
  mother_age_2010 + years_passed = 2 * (lucy_age_2010 + years_passed) :=
by sorry

end NUMINAMATH_CALUDE_lucy_mother_age_relation_l149_14913


namespace NUMINAMATH_CALUDE_abc_product_l149_14988

theorem abc_product (a b c : ℝ) 
  (eq1 : a + b = 23)
  (eq2 : b + c = 25)
  (eq3 : c + a = 30) :
  a * b * c = 2016 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l149_14988


namespace NUMINAMATH_CALUDE_triangle_angles_l149_14920

theorem triangle_angles (a b c : ℝ) (h1 : a = 3) (h2 : b = Real.sqrt 8) (h3 : c = 2 + Real.sqrt 2) :
  ∃ (θ φ ψ : ℝ),
    Real.cos θ = (10 + Real.sqrt 2) / 18 ∧
    Real.cos φ = (11 - 4 * Real.sqrt 2) / (12 * Real.sqrt 2) ∧
    θ + φ + ψ = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_l149_14920


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt35_l149_14949

theorem closest_integer_to_sqrt35 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 35| ≤ |m - Real.sqrt 35| ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt35_l149_14949


namespace NUMINAMATH_CALUDE_acute_triangle_cotangent_sum_range_l149_14921

theorem acute_triangle_cotangent_sum_range (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  b^2 - a^2 = a * c →  -- Given condition
  1 < 1 / Real.tan A + 1 / Real.tan B ∧ 
  1 / Real.tan A + 1 / Real.tan B < 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_cotangent_sum_range_l149_14921


namespace NUMINAMATH_CALUDE_tobacco_acreage_increase_l149_14942

/-- Calculates the increase in tobacco acreage when changing crop ratios -/
theorem tobacco_acreage_increase (total_land : ℝ) (initial_ratio_tobacco : ℝ) 
  (initial_ratio_total : ℝ) (new_ratio_tobacco : ℝ) (new_ratio_total : ℝ) :
  total_land = 1350 ∧ 
  initial_ratio_tobacco = 2 ∧ 
  initial_ratio_total = 9 ∧ 
  new_ratio_tobacco = 5 ∧ 
  new_ratio_total = 9 →
  (new_ratio_tobacco / new_ratio_total - initial_ratio_tobacco / initial_ratio_total) * total_land = 450 :=
by sorry

end NUMINAMATH_CALUDE_tobacco_acreage_increase_l149_14942


namespace NUMINAMATH_CALUDE_shaded_fraction_is_one_twelfth_l149_14980

/-- A point in a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A rectangle defined by its top-left and bottom-right corners -/
structure Rectangle where
  topLeft : GridPoint
  bottomRight : GridPoint

/-- The 6x6 grid -/
def gridSize : ℕ := 6

/-- The rectangle in question -/
def shadedRectangle : Rectangle := {
  topLeft := { x := 2, y := 5 }
  bottomRight := { x := 3, y := 2 }
}

/-- Calculate the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ :=
  (r.bottomRight.x - r.topLeft.x) * (r.topLeft.y - r.bottomRight.y)

/-- Calculate the area of the entire grid -/
def gridArea : ℕ := gridSize * gridSize

/-- The fraction of the grid occupied by the shaded rectangle -/
def shadedFraction : ℚ :=
  (rectangleArea shadedRectangle : ℚ) / gridArea

/-- Theorem: The shaded fraction is equal to 1/12 -/
theorem shaded_fraction_is_one_twelfth : shadedFraction = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_is_one_twelfth_l149_14980


namespace NUMINAMATH_CALUDE_cube_sum_over_product_is_18_l149_14940

theorem cube_sum_over_product_is_18 
  (a b c : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_15 : a + b + c = 15)
  (squared_diff_sum : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_is_18_l149_14940


namespace NUMINAMATH_CALUDE_gcd_difference_theorem_l149_14925

theorem gcd_difference_theorem : Nat.gcd 5610 210 - 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_gcd_difference_theorem_l149_14925


namespace NUMINAMATH_CALUDE_matching_socks_probability_l149_14963

/-- The number of blue-bottomed socks -/
def blue_socks : ℕ := 12

/-- The number of red-bottomed socks -/
def red_socks : ℕ := 10

/-- The number of green-bottomed socks -/
def green_socks : ℕ := 6

/-- The total number of socks -/
def total_socks : ℕ := blue_socks + red_socks + green_socks

/-- The number of ways to choose 2 socks from n socks -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of picking a matching pair of socks -/
theorem matching_socks_probability : 
  (choose_two blue_socks + choose_two red_socks + choose_two green_socks) / choose_two total_socks = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_matching_socks_probability_l149_14963


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l149_14975

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 418 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l149_14975


namespace NUMINAMATH_CALUDE_tangent_circles_ratio_l149_14982

/-- Two circles touching internally with specific tangent properties -/
structure TangentCircles where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  touch_internally : R > r  -- Circles touch internally
  radii_angle : ℝ  -- Angle between the two radii of the larger circle
  radii_tangent : Bool  -- The two radii are tangent to the smaller circle

/-- Theorem stating the ratio of radii for circles with specific tangent properties -/
theorem tangent_circles_ratio 
  (c : TangentCircles) 
  (h1 : c.radii_angle = 60) 
  (h2 : c.radii_tangent = true) : 
  c.R / c.r = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_ratio_l149_14982


namespace NUMINAMATH_CALUDE_leapYearsIn123Years_l149_14967

/-- In a calendrical system where leap years occur every three years, 
    this function calculates the number of leap years in a given period. -/
def leapYearsCount (periodLength : ℕ) : ℕ :=
  periodLength / 3

/-- Theorem stating that in a 123-year period, the number of leap years is 41. -/
theorem leapYearsIn123Years : leapYearsCount 123 = 41 := by
  sorry

end NUMINAMATH_CALUDE_leapYearsIn123Years_l149_14967


namespace NUMINAMATH_CALUDE_online_price_calculation_l149_14985

/-- Calculates the price a buyer observes online for a product sold by a distributor through an online store, given various costs and desired profit margin. -/
theorem online_price_calculation 
  (producer_price : ℝ) 
  (shipping_cost : ℝ) 
  (commission_rate : ℝ) 
  (tax_rate : ℝ) 
  (profit_margin : ℝ) 
  (h1 : producer_price = 19) 
  (h2 : shipping_cost = 5) 
  (h3 : commission_rate = 0.2) 
  (h4 : tax_rate = 0.1) 
  (h5 : profit_margin = 0.2) : 
  ∃ (online_price : ℝ), online_price = 39.6 ∧ 
  online_price * (1 - commission_rate) = 
    (producer_price + shipping_cost) * (1 + profit_margin) * (1 + tax_rate) := by
  sorry

end NUMINAMATH_CALUDE_online_price_calculation_l149_14985


namespace NUMINAMATH_CALUDE_divisibility_by_three_l149_14968

theorem divisibility_by_three (u v : ℤ) (h : (9 : ℤ) ∣ (u^2 + u*v + v^2)) : (3 : ℤ) ∣ u ∧ (3 : ℤ) ∣ v := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l149_14968


namespace NUMINAMATH_CALUDE_andrews_cat_catch_l149_14956

theorem andrews_cat_catch (martha_cat cara_cat T : ℕ) : 
  martha_cat = 10 →
  cara_cat = 47 →
  T = martha_cat + cara_cat →
  T^2 + 2 = 3251 :=
by
  sorry

end NUMINAMATH_CALUDE_andrews_cat_catch_l149_14956


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l149_14943

theorem quadratic_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l149_14943


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l149_14938

/-- The y-intercept of the line 4x + 7y - 3xy = 28 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) : 
  (4 * x + 7 * y - 3 * x * y = 28) → 
  (x = 0 → y = 4) :=
by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l149_14938


namespace NUMINAMATH_CALUDE_owen_sleep_time_l149_14983

/-- Owen's daily schedule and sleep time calculation -/
theorem owen_sleep_time :
  let hours_in_day : ℝ := 24
  let work_hours : ℝ := 6
  let commute_hours : ℝ := 2
  let exercise_hours : ℝ := 3
  let cooking_hours : ℝ := 1
  let leisure_hours : ℝ := 3
  let grooming_hours : ℝ := 1.5
  let total_activity_hours := work_hours + commute_hours + exercise_hours + 
                              cooking_hours + leisure_hours + grooming_hours
  let sleep_hours := hours_in_day - total_activity_hours
  sleep_hours = 7.5 := by sorry

end NUMINAMATH_CALUDE_owen_sleep_time_l149_14983


namespace NUMINAMATH_CALUDE_last_segment_speed_l149_14989

/-- Represents the average speed during a journey segment -/
structure JourneySegment where
  duration : ℚ  -- Duration in hours
  speed : ℚ     -- Average speed in mph
  distance : ℚ  -- Distance traveled in miles

/-- Represents a complete journey -/
structure Journey where
  totalDistance : ℚ
  totalTime : ℚ
  segments : List JourneySegment

/-- Calculates the average speed for a given distance and time -/
def averageSpeed (distance : ℚ) (time : ℚ) : ℚ :=
  distance / time

theorem last_segment_speed (j : Journey) 
  (h1 : j.totalDistance = 120)
  (h2 : j.totalTime = 2)
  (h3 : j.segments.length = 3)
  (h4 : j.segments[0].duration = 2/3)
  (h5 : j.segments[0].speed = 50)
  (h6 : j.segments[1].duration = 5/6)
  (h7 : j.segments[1].speed = 60)
  (h8 : j.segments[2].duration = 1/2) :
  averageSpeed j.segments[2].distance j.segments[2].duration = 220/3 := by
  sorry

#eval (220 : ℚ) / 3  -- To verify the result is approximately 73.33

end NUMINAMATH_CALUDE_last_segment_speed_l149_14989


namespace NUMINAMATH_CALUDE_no_integer_root_2016_l149_14996

theorem no_integer_root_2016 (a b c d : ℤ) (p : ℤ → ℤ) :
  (∀ x : ℤ, p x = a * x^3 + b * x^2 + c * x + d) →
  p 1 = 2015 →
  p 2 = 2017 →
  ∀ x : ℤ, p x ≠ 2016 := by
sorry

end NUMINAMATH_CALUDE_no_integer_root_2016_l149_14996


namespace NUMINAMATH_CALUDE_ali_flower_sales_l149_14910

def monday_sales : ℕ := 4
def tuesday_sales : ℕ := 8
def wednesday_sales : ℕ := monday_sales + 3
def thursday_sales : ℕ := 6
def friday_sales : ℕ := 2 * monday_sales
def saturday_bundles : ℕ := 5
def flowers_per_bundle : ℕ := 9
def saturday_sales : ℕ := saturday_bundles * flowers_per_bundle

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales

theorem ali_flower_sales : total_sales = 78 := by
  sorry

end NUMINAMATH_CALUDE_ali_flower_sales_l149_14910


namespace NUMINAMATH_CALUDE_range_of_a_l149_14977

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x a : ℝ) : Prop := abs x > a

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a))

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (∀ x, p x → q x a) ∧ necessary_not_sufficient a ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l149_14977
