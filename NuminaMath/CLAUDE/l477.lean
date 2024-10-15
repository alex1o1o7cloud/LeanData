import Mathlib

namespace NUMINAMATH_CALUDE_max_cd_length_l477_47710

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    where c = 4 and CD⊥AB, prove that the maximum value of CD is 2√3 under the given condition. -/
theorem max_cd_length (a b : ℝ) (A B C : ℝ) :
  let c : ℝ := 4
  (c * Real.cos C * Real.cos (A - B) + c = c * Real.sin C ^ 2 + b * Real.sin A * Real.sin C) →
  (∃ (D : ℝ), D ≤ 2 * Real.sqrt 3 ∧
    ∀ (E : ℝ), (c * Real.cos C * Real.cos (A - B) + c = c * Real.sin C ^ 2 + b * Real.sin A * Real.sin C) →
      E ≤ D) :=
by sorry

end NUMINAMATH_CALUDE_max_cd_length_l477_47710


namespace NUMINAMATH_CALUDE_suit_price_calculation_l477_47747

def original_price : ℚ := 200
def increase_rate : ℚ := 0.30
def discount_rate : ℚ := 0.30
def tax_rate : ℚ := 0.07

def increased_price : ℚ := original_price * (1 + increase_rate)
def discounted_price : ℚ := increased_price * (1 - discount_rate)
def final_price : ℚ := discounted_price * (1 + tax_rate)

theorem suit_price_calculation :
  final_price = 194.74 := by sorry

end NUMINAMATH_CALUDE_suit_price_calculation_l477_47747


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l477_47739

theorem smallest_solution_quadratic :
  ∃ (x : ℝ), x = 2/3 ∧ 6*x^2 - 19*x + 10 = 0 ∧ ∀ (y : ℝ), 6*y^2 - 19*y + 10 = 0 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l477_47739


namespace NUMINAMATH_CALUDE_equation_system_solution_l477_47774

theorem equation_system_solution (n p : ℕ) :
  (∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x + p * y = n ∧ x + y = p^2) ↔
  (p > 1 ∧ (n - 1) % (p - 1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l477_47774


namespace NUMINAMATH_CALUDE_equation_solution_solution_set_l477_47740

theorem equation_solution (x : ℝ) : 
  x ≠ 7 → (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21) :=
by sorry

theorem solution_set : 
  {x : ℝ | (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21)} = {x : ℝ | x ≠ 7} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_solution_set_l477_47740


namespace NUMINAMATH_CALUDE_football_tournament_score_product_l477_47767

/-- Represents a football team's score in the tournament -/
structure TeamScore where
  points : ℕ

/-- Represents the scores of all teams in the tournament -/
structure TournamentResult where
  scores : Finset TeamScore
  team_count : ℕ
  is_round_robin : Bool
  consecutive_scores : Bool

/-- The main theorem about the tournament results -/
theorem football_tournament_score_product (result : TournamentResult) :
  result.team_count = 4 ∧
  result.is_round_robin = true ∧
  result.consecutive_scores = true ∧
  result.scores.card = 4 →
  (result.scores.toList.map (λ s => s.points)).prod = 120 := by
  sorry

end NUMINAMATH_CALUDE_football_tournament_score_product_l477_47767


namespace NUMINAMATH_CALUDE_drink_conversion_l477_47777

theorem drink_conversion (x : ℚ) : 
  (4 / (4 + x) * 63 = 3 / 7 * (63 + 21)) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_drink_conversion_l477_47777


namespace NUMINAMATH_CALUDE_geometric_sequence_product_relation_l477_47705

/-- Represents a geometric sequence with 3n terms -/
structure GeometricSequence (α : Type*) [CommRing α] where
  n : ℕ
  terms : Fin (3 * n) → α
  is_geometric : ∀ i j k, i < j → j < k → terms j ^ 2 = terms i * terms k

/-- The product of n consecutive terms in a geometric sequence -/
def product_n_terms {α : Type*} [CommRing α] (seq : GeometricSequence α) (start : ℕ) : α :=
  (List.range seq.n).foldl (λ acc i => acc * seq.terms ⟨start * seq.n + i, sorry⟩) 1

/-- Theorem: In a geometric sequence with 3n terms, if A is the product of the first n terms,
    B is the product of the next n terms, and C is the product of the last n terms, then AC = B² -/
theorem geometric_sequence_product_relation {α : Type*} [CommRing α] (seq : GeometricSequence α) :
  let A := product_n_terms seq 0
  let B := product_n_terms seq 1
  let C := product_n_terms seq 2
  A * C = B ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_relation_l477_47705


namespace NUMINAMATH_CALUDE_intersection_locus_is_ellipse_l477_47745

/-- The locus of points (x, y) satisfying a system of equations forms an ellipse -/
theorem intersection_locus_is_ellipse :
  ∀ (s x y : ℝ), 
  (2 * s * x - 3 * y - 4 * s = 0) → 
  (x - 3 * s * y + 4 = 0) → 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_intersection_locus_is_ellipse_l477_47745


namespace NUMINAMATH_CALUDE_box_stacking_comparison_l477_47744

/-- Represents the height of a stack of boxes -/
def stack_height (box_height : ℝ) (num_floors : ℕ) : ℝ :=
  box_height * (num_floors : ℝ)

/-- The problem statement -/
theorem box_stacking_comparison : 
  let box_a_height : ℝ := 3
  let box_b_height : ℝ := 3.5
  let taehyung_floors : ℕ := 16
  let yoongi_floors : ℕ := 14
  
  stack_height box_b_height yoongi_floors - stack_height box_a_height taehyung_floors = 1 := by
  sorry

end NUMINAMATH_CALUDE_box_stacking_comparison_l477_47744


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l477_47768

/-- Represents a hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Represents a chord PQ -/
structure Chord where
  P : ℝ × ℝ
  Q : ℝ × ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Checks if a chord is perpendicular to the real axis -/
def is_perpendicular_to_real_axis (c : Chord) : Prop := sorry

/-- Checks if a chord passes through a given point -/
def passes_through (c : Chord) (p : ℝ × ℝ) : Prop := sorry

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_eccentricity_theorem (h : Hyperbola) (c : Chord) :
  is_perpendicular_to_real_axis c →
  passes_through c h.F₂ →
  angle c.P h.F₁ c.Q = π / 2 →
  eccentricity h = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l477_47768


namespace NUMINAMATH_CALUDE_not_all_same_color_probability_l477_47738

def num_people : ℕ := 3
def num_colors : ℕ := 5

theorem not_all_same_color_probability :
  (num_colors^num_people - num_colors) / num_colors^num_people = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_not_all_same_color_probability_l477_47738


namespace NUMINAMATH_CALUDE_train_length_proof_l477_47753

-- Define the given conditions
def faster_train_speed : ℝ := 42
def slower_train_speed : ℝ := 36
def passing_time : ℝ := 36

-- Define the theorem
theorem train_length_proof :
  let relative_speed := faster_train_speed - slower_train_speed
  let speed_in_mps := relative_speed * (5 / 18)
  let distance := speed_in_mps * passing_time
  let train_length := distance / 2
  train_length = 30 := by sorry

end NUMINAMATH_CALUDE_train_length_proof_l477_47753


namespace NUMINAMATH_CALUDE_abc_acute_angle_implies_m_values_l477_47723

def OA : Fin 2 → ℝ := ![3, -4]
def OB : Fin 2 → ℝ := ![6, -3]
def OC (m : ℝ) : Fin 2 → ℝ := ![5 - m, -3 - m]

def BA : Fin 2 → ℝ := ![OA 0 - OB 0, OA 1 - OB 1]
def BC (m : ℝ) : Fin 2 → ℝ := ![OC m 0 - OB 0, OC m 1 - OB 1]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1)

def is_acute_angle (m : ℝ) : Prop := dot_product BA (BC m) > 0

theorem abc_acute_angle_implies_m_values :
  ∀ m : ℝ, is_acute_angle m → (m = 0 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_abc_acute_angle_implies_m_values_l477_47723


namespace NUMINAMATH_CALUDE_pad_pages_proof_l477_47787

theorem pad_pages_proof (P : ℝ) 
  (h1 : P - (0.25 * P + 10) = 80) : P = 120 := by
  sorry

end NUMINAMATH_CALUDE_pad_pages_proof_l477_47787


namespace NUMINAMATH_CALUDE_part_a_part_b_l477_47716

-- Define the set M of functions satisfying the given conditions
def M : Set (ℤ → ℝ) :=
  {f | f 0 ≠ 0 ∧ ∀ n m : ℤ, f n * f m = f (n + m) + f (n - m)}

-- Theorem for part (a)
theorem part_a (f : ℤ → ℝ) (hf : f ∈ M) (h1 : f 1 = 5/2) :
  ∀ n : ℤ, f n = 2^n + 2^(-n) := by sorry

-- Theorem for part (b)
theorem part_b (f : ℤ → ℝ) (hf : f ∈ M) (h1 : f 1 = Real.sqrt 3) :
  ∀ n : ℤ, f n = 2 * Real.cos (π * n / 6) := by sorry

end NUMINAMATH_CALUDE_part_a_part_b_l477_47716


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l477_47728

theorem polynomial_remainder_theorem : ∃ q : Polynomial ℝ, 
  3 * X^3 + 2 * X^2 - 20 * X + 47 = (X - 3) * q + 86 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l477_47728


namespace NUMINAMATH_CALUDE_total_games_calculation_l477_47772

/-- The number of baseball games played at night -/
def night_games : ℕ := 128

/-- The number of games Joan attended -/
def attended_games : ℕ := 395

/-- The number of games Joan missed -/
def missed_games : ℕ := 469

/-- The total number of baseball games played this year -/
def total_games : ℕ := attended_games + missed_games

theorem total_games_calculation : 
  total_games = attended_games + missed_games := by sorry

end NUMINAMATH_CALUDE_total_games_calculation_l477_47772


namespace NUMINAMATH_CALUDE_second_crew_tractors_second_crew_is_seven_l477_47719

/-- Calculates the number of tractors in the second crew given the farming conditions --/
theorem second_crew_tractors (total_acres : ℕ) (total_days : ℕ) (first_crew_tractors : ℕ) 
  (first_crew_days : ℕ) (second_crew_days : ℕ) (acres_per_day : ℕ) : ℕ :=
  let first_crew_acres := first_crew_tractors * first_crew_days * acres_per_day
  let remaining_acres := total_acres - first_crew_acres
  let acres_per_tractor := second_crew_days * acres_per_day
  remaining_acres / acres_per_tractor

/-- Proves that the number of tractors in the second crew is 7 --/
theorem second_crew_is_seven : 
  second_crew_tractors 1700 5 2 2 3 68 = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_crew_tractors_second_crew_is_seven_l477_47719


namespace NUMINAMATH_CALUDE_roger_bike_distance_l477_47725

def morning_distance : ℝ := 2
def evening_multiplier : ℝ := 5

theorem roger_bike_distance : 
  morning_distance + evening_multiplier * morning_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_roger_bike_distance_l477_47725


namespace NUMINAMATH_CALUDE_monica_books_next_year_l477_47727

/-- The number of books Monica read last year -/
def books_last_year : ℕ := 16

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 2 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 2 * books_this_year + 5

/-- Theorem stating the number of books Monica will read next year -/
theorem monica_books_next_year : books_next_year = 69 := by
  sorry

end NUMINAMATH_CALUDE_monica_books_next_year_l477_47727


namespace NUMINAMATH_CALUDE_letter_placement_l477_47786

theorem letter_placement (n_letters : ℕ) (n_boxes : ℕ) : n_letters = 3 ∧ n_boxes = 5 → n_boxes ^ n_letters = 125 := by
  sorry

end NUMINAMATH_CALUDE_letter_placement_l477_47786


namespace NUMINAMATH_CALUDE_club_sports_theorem_l477_47718

/-- The number of people who do not play a sport in a club -/
def people_not_playing (total : ℕ) (tennis : ℕ) (baseball : ℕ) (both : ℕ) : ℕ :=
  total - (tennis + baseball - both)

/-- Theorem: In a club with 310 people, where 138 play tennis, 255 play baseball, 
    and 94 play both sports, 11 people do not play a sport. -/
theorem club_sports_theorem : people_not_playing 310 138 255 94 = 11 := by
  sorry

end NUMINAMATH_CALUDE_club_sports_theorem_l477_47718


namespace NUMINAMATH_CALUDE_double_base_exponent_equality_l477_47708

theorem double_base_exponent_equality (a b x : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  (2 * a) ^ (2 * b) = a ^ b * x ^ 3 → x = (4 ^ b * a ^ b) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_double_base_exponent_equality_l477_47708


namespace NUMINAMATH_CALUDE_a_eq_zero_necessary_not_sufficient_l477_47735

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem a_eq_zero_necessary_not_sufficient :
  ∀ (z : ℂ), 
  (IsPurelyImaginary z → z.re = 0) ∧ 
  ∃ (z : ℂ), z.re = 0 ∧ ¬IsPurelyImaginary z :=
by sorry

end NUMINAMATH_CALUDE_a_eq_zero_necessary_not_sufficient_l477_47735


namespace NUMINAMATH_CALUDE_combined_instruments_count_l477_47765

/-- Represents the number of instruments owned by a person -/
structure InstrumentCount where
  flutes : ℕ
  horns : ℕ
  harps : ℕ

/-- Calculates the total number of instruments -/
def totalInstruments (ic : InstrumentCount) : ℕ :=
  ic.flutes + ic.horns + ic.harps

/-- Charlie's instrument count -/
def charlie : InstrumentCount :=
  { flutes := 1, horns := 2, harps := 1 }

/-- Carli's instrument count -/
def carli : InstrumentCount :=
  { flutes := 2 * charlie.flutes,
    horns := charlie.horns / 2,
    harps := 0 }

/-- Theorem: The combined total number of musical instruments owned by Charlie and Carli is 7 -/
theorem combined_instruments_count :
  totalInstruments charlie + totalInstruments carli = 7 := by
  sorry

end NUMINAMATH_CALUDE_combined_instruments_count_l477_47765


namespace NUMINAMATH_CALUDE_divisible_by_101_exists_l477_47724

theorem divisible_by_101_exists (n : ℕ) (h : n ≥ 10^2018) : 
  ∃ k : ℕ, ∃ m : ℕ, m ≥ n ∧ m = n + k ∧ m % 101 = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_101_exists_l477_47724


namespace NUMINAMATH_CALUDE_probability_diamond_then_face_correct_l477_47717

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
| hearts | diamonds | clubs | spades

/-- Represents the rank of a card -/
inductive Rank
| two | three | four | five | six | seven | eight | nine | ten
| jack | queen | king | ace

/-- Represents a playing card -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- Checks if a card is a diamond -/
def is_diamond (c : Card) : Prop :=
  c.suit = Suit.diamonds

/-- Checks if a card is a face card -/
def is_face_card (c : Card) : Prop :=
  c.rank = Rank.jack ∨ c.rank = Rank.queen ∨ c.rank = Rank.king

/-- The number of diamonds in a standard deck -/
def diamond_count : Nat := 13

/-- The number of face cards in a standard deck -/
def face_card_count : Nat := 12

/-- The probability of drawing a diamond as the first card and a face card as the second card -/
def probability_diamond_then_face (d : Deck) : ℚ :=
  47 / 884

theorem probability_diamond_then_face_correct (d : Deck) :
  probability_diamond_then_face d = 47 / 884 :=
sorry

end NUMINAMATH_CALUDE_probability_diamond_then_face_correct_l477_47717


namespace NUMINAMATH_CALUDE_cos_sin_identity_l477_47769

theorem cos_sin_identity : 
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.sin (66 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l477_47769


namespace NUMINAMATH_CALUDE_coefficient_of_x_l477_47736

theorem coefficient_of_x (a x y : ℝ) : 
  a * x + y = 19 →
  x + 3 * y = 1 →
  3 * x + 2 * y = 10 →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l477_47736


namespace NUMINAMATH_CALUDE_negative_three_times_five_l477_47706

theorem negative_three_times_five : (-3 : ℤ) * 5 = -15 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_five_l477_47706


namespace NUMINAMATH_CALUDE_inverse_function_problem_l477_47757

theorem inverse_function_problem (g : ℝ → ℝ) (g_inv : ℝ → ℝ) 
  (h1 : Function.LeftInverse g_inv g) 
  (h2 : Function.RightInverse g_inv g)
  (h3 : g 4 = 6)
  (h4 : g 6 = 2)
  (h5 : g 3 = 7) :
  g_inv (g_inv 7 + g_inv 6) = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l477_47757


namespace NUMINAMATH_CALUDE_infinitely_many_integer_pairs_l477_47707

theorem infinitely_many_integer_pairs : 
  ∃ (S : Set (ℤ × ℤ)), Set.Infinite S ∧ 
    ∀ (pair : ℤ × ℤ), pair ∈ S → 
      ∃ (k : ℤ), (pair.1 + 1) / pair.2 + (pair.2 + 1) / pair.1 = k :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_integer_pairs_l477_47707


namespace NUMINAMATH_CALUDE_not_right_triangle_4_6_8_l477_47762

/-- Checks if three line segments can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Theorem stating that the line segments 4, 6, and 8 cannot form a right triangle -/
theorem not_right_triangle_4_6_8 : ¬ is_right_triangle 4 6 8 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_4_6_8_l477_47762


namespace NUMINAMATH_CALUDE_pre_bought_tickets_l477_47729

/-- The number of people who pre-bought plane tickets -/
def num_pre_buyers : ℕ := 20

/-- The price of a pre-bought ticket -/
def pre_bought_price : ℕ := 155

/-- The price of a ticket bought at the gate -/
def gate_price : ℕ := 200

/-- The number of people who bought tickets at the gate -/
def num_gate_buyers : ℕ := 30

/-- The difference in total amount paid between gate buyers and pre-buyers -/
def price_difference : ℕ := 2900

theorem pre_bought_tickets : 
  num_pre_buyers * pre_bought_price + price_difference = num_gate_buyers * gate_price := by
  sorry

end NUMINAMATH_CALUDE_pre_bought_tickets_l477_47729


namespace NUMINAMATH_CALUDE_smallest_prime_ten_less_than_square_l477_47792

theorem smallest_prime_ten_less_than_square : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > 0 ∧ Nat.Prime m ∧ (∃ (k : ℕ), m = k^2 - 10) → n ≤ m) ∧
  n > 0 ∧ Nat.Prime n ∧ (∃ (k : ℕ), n = k^2 - 10) ∧ n = 71 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_ten_less_than_square_l477_47792


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l477_47775

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 4}

theorem complement_of_A_in_U : 
  (U \ A) = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l477_47775


namespace NUMINAMATH_CALUDE_polynomial_intercept_nonzero_coeff_l477_47715

theorem polynomial_intercept_nonzero_coeff 
  (a b c d e f : ℝ) 
  (Q : ℝ → ℝ) 
  (h_Q : ∀ x, Q x = x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) 
  (h_roots : ∃ p q r s t : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0 ∧ 
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
    q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
    r ≠ s ∧ r ≠ t ∧ 
    s ≠ t ∧
    Q p = 0 ∧ Q q = 0 ∧ Q r = 0 ∧ Q s = 0 ∧ Q t = 0)
  (h_zero_root : Q 0 = 0) :
  d ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_intercept_nonzero_coeff_l477_47715


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l477_47761

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem P_sufficient_not_necessary_for_Q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l477_47761


namespace NUMINAMATH_CALUDE_min_sum_inverse_squares_min_sum_inverse_squares_value_min_sum_inverse_squares_equality_l477_47764

theorem min_sum_inverse_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c → a^2 + b^2 + c^2 = 1 →
  1/x^2 + 1/y^2 + 1/z^2 ≤ 1/a^2 + 1/b^2 + 1/c^2 :=
by sorry

theorem min_sum_inverse_squares_value (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  1/x^2 + 1/y^2 + 1/z^2 ≥ 9 :=
by sorry

theorem min_sum_inverse_squares_equality :
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 
  x^2 + y^2 + z^2 = 1 ∧ 1/x^2 + 1/y^2 + 1/z^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_inverse_squares_min_sum_inverse_squares_value_min_sum_inverse_squares_equality_l477_47764


namespace NUMINAMATH_CALUDE_possible_m_values_l477_47776

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem possible_m_values : 
  {m : ℝ | B m ⊆ A} = {-1/2, 0, 1/3} := by sorry

end NUMINAMATH_CALUDE_possible_m_values_l477_47776


namespace NUMINAMATH_CALUDE_jake_weight_proof_l477_47734

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 198

/-- Kendra's weight in pounds -/
def kendra_weight : ℝ := 95

/-- The sum of Jake's and Kendra's weights -/
def total_weight : ℝ := 293

theorem jake_weight_proof :
  (jake_weight - 8 = 2 * kendra_weight) ∧
  (jake_weight + kendra_weight = total_weight) →
  jake_weight = 198 := by
sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l477_47734


namespace NUMINAMATH_CALUDE_inequality_theorem_l477_47784

theorem inequality_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  2 * Real.log m - n > 2 * Real.log n - m := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l477_47784


namespace NUMINAMATH_CALUDE_average_income_calculation_l477_47750

/-- Given the average monthly incomes of pairs of individuals and the income of one individual,
    calculate the average monthly income of a specific pair. -/
theorem average_income_calculation (P Q R : ℕ) : 
  (P + Q) / 2 = 2050 →
  (Q + R) / 2 = 5250 →
  P = 3000 →
  (P + R) / 2 = 6200 := by
sorry

end NUMINAMATH_CALUDE_average_income_calculation_l477_47750


namespace NUMINAMATH_CALUDE_ramen_bread_intersection_l477_47742

theorem ramen_bread_intersection (total : ℕ) (ramen : ℕ) (bread : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : ramen = 289)
  (h3 : bread = 337)
  (h4 : neither = 56) :
  ramen + bread - total + neither = 182 :=
by sorry

end NUMINAMATH_CALUDE_ramen_bread_intersection_l477_47742


namespace NUMINAMATH_CALUDE_adam_shopping_cost_l477_47783

/-- The total cost of Adam's shopping given the number of sandwiches, cost per sandwich, and cost of water. -/
def total_cost (num_sandwiches : ℕ) (sandwich_price : ℕ) (water_price : ℕ) : ℕ :=
  num_sandwiches * sandwich_price + water_price

/-- Theorem stating that Adam's total shopping cost is $11. -/
theorem adam_shopping_cost :
  total_cost 3 3 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_adam_shopping_cost_l477_47783


namespace NUMINAMATH_CALUDE_min_black_edges_is_four_l477_47770

/-- Represents the coloring of a cube's edges -/
structure CubeColoring where
  edges : Fin 12 → Bool  -- True represents black, False represents red

/-- Checks if a face has an even number of black edges -/
def has_even_black_edges (c : CubeColoring) (face : Fin 6) : Bool :=
  sorry

/-- Checks if all faces have an even number of black edges -/
def all_faces_even_black (c : CubeColoring) : Prop :=
  ∀ face : Fin 6, has_even_black_edges c face

/-- Counts the number of black edges in a coloring -/
def count_black_edges (c : CubeColoring) : Nat :=
  sorry

/-- The main theorem: The minimum number of black edges required is 4 -/
theorem min_black_edges_is_four :
  (∃ c : CubeColoring, all_faces_even_black c ∧ count_black_edges c = 4) ∧
  (∀ c : CubeColoring, all_faces_even_black c → count_black_edges c ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_min_black_edges_is_four_l477_47770


namespace NUMINAMATH_CALUDE_hyperbola_equation_l477_47746

/-- Given a hyperbola with the general equation y²/a² - x²/b² = 1 where a > 0 and b > 0,
    an asymptote equation of 3x + 4y = 0, and a focus at (0,5),
    prove that the specific equation of the hyperbola is y²/9 - x²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote : ∀ x y : ℝ, 3 * x + 4 * y = 0 → (y / x = -3 / 4 ∨ y / x = 3 / 4))
  (focus : (0 : ℝ) ^ 2 + 5 ^ 2 = (a ^ 2 + b ^ 2)) :
  ∀ x y : ℝ, y ^ 2 / 9 - x ^ 2 / 16 = 1 ↔ y ^ 2 / a ^ 2 - x ^ 2 / b ^ 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l477_47746


namespace NUMINAMATH_CALUDE_star_polygon_n_value_l477_47713

/-- A regular star polygon with n points, where each point has two angles. -/
structure StarPolygon where
  n : ℕ
  angle_C : ℝ
  angle_D : ℝ

/-- Properties of the star polygon -/
def is_valid_star_polygon (s : StarPolygon) : Prop :=
  s.n > 0 ∧
  s.angle_C > 0 ∧
  s.angle_D > 0 ∧
  s.angle_C = s.angle_D - 15 ∧
  s.n * 15 = 360

theorem star_polygon_n_value (s : StarPolygon) (h : is_valid_star_polygon s) : s.n = 24 := by
  sorry

#check star_polygon_n_value

end NUMINAMATH_CALUDE_star_polygon_n_value_l477_47713


namespace NUMINAMATH_CALUDE_f_decreasing_after_seven_fourths_l477_47751

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x < 1 then 2 * x^2 - x + 1
  else -2 * x^2 + 7 * x - 7

-- State the theorem
theorem f_decreasing_after_seven_fourths :
  (∀ x, f (x + 1) = -f (-(x + 1))) →
  (∀ x < 1, f x = 2 * x^2 - x + 1) →
  ∀ x > (7/4 : ℝ), ∀ y > x, f y < f x :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_after_seven_fourths_l477_47751


namespace NUMINAMATH_CALUDE_volume_of_midpoint_set_l477_47752

/-- A regular tetrahedron -/
structure RegularTetrahedron :=
  (edge_length : ℝ)

/-- The set of midpoints of segments whose endpoints belong to different tetrahedra -/
def midpoint_set (t1 t2 : RegularTetrahedron) : Set (Fin 3 → ℝ) :=
  sorry

/-- The volume of a set in ℝ³ -/
noncomputable def volume (s : Set (Fin 3 → ℝ)) : ℝ :=
  sorry

/-- Central symmetry transformation -/
def central_symmetry (t : RegularTetrahedron) : RegularTetrahedron :=
  sorry

theorem volume_of_midpoint_set :
  ∀ t : RegularTetrahedron,
  t.edge_length = Real.sqrt 2 →
  volume (midpoint_set t (central_symmetry t)) = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_volume_of_midpoint_set_l477_47752


namespace NUMINAMATH_CALUDE_probability_green_second_is_three_fifths_l477_47794

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  white : ℕ := 0
  black : ℕ := 0
  red : ℕ := 0
  green : ℕ := 0

/-- Calculate the probability of drawing a green marble as the second marble -/
def probabilityGreenSecond (bagX bagY bagZ : BagContents) : ℚ :=
  let probWhiteX := bagX.white / (bagX.white + bagX.black)
  let probBlackX := bagX.black / (bagX.white + bagX.black)
  let probGreenY := bagY.green / (bagY.red + bagY.green)
  let probGreenZ := bagZ.green / (bagZ.red + bagZ.green)
  probWhiteX * probGreenY + probBlackX * probGreenZ

/-- The main theorem to prove -/
theorem probability_green_second_is_three_fifths :
  let bagX := BagContents.mk 5 5 0 0
  let bagY := BagContents.mk 0 0 7 8
  let bagZ := BagContents.mk 0 0 3 6
  probabilityGreenSecond bagX bagY bagZ = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_probability_green_second_is_three_fifths_l477_47794


namespace NUMINAMATH_CALUDE_vending_machine_failure_rate_l477_47760

/-- Calculates the failure rate of a vending machine. -/
theorem vending_machine_failure_rate 
  (total_users : ℕ) 
  (snacks_dropped : ℕ) 
  (extra_snack_rate : ℚ) : 
  total_users = 30 → 
  snacks_dropped = 28 → 
  extra_snack_rate = 1/10 → 
  (total_users : ℚ) - snacks_dropped = 
    total_users * (1 - extra_snack_rate) * (1/6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_vending_machine_failure_rate_l477_47760


namespace NUMINAMATH_CALUDE_total_sum_is_71_rupees_l477_47749

/-- Calculates the total sum of money in rupees given the number of 20 paise and 25 paise coins -/
def total_sum_in_rupees (total_coins : ℕ) (coins_20_paise : ℕ) : ℚ :=
  let coins_25_paise := total_coins - coins_20_paise
  let value_20_paise := 20 * coins_20_paise
  let value_25_paise := 25 * coins_25_paise
  (value_20_paise + value_25_paise : ℚ) / 100

/-- Theorem stating that given 342 total coins with 290 being 20 paise coins, 
    the total sum of money is 71 rupees -/
theorem total_sum_is_71_rupees :
  total_sum_in_rupees 342 290 = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_is_71_rupees_l477_47749


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_three_l477_47759

theorem subset_implies_a_equals_three (A B : Set ℕ) (a : ℕ) 
  (h1 : A = {1, 3})
  (h2 : B = {1, 2, a})
  (h3 : A ⊆ B) : 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_three_l477_47759


namespace NUMINAMATH_CALUDE_correct_reasoning_directions_l477_47793

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| SpecificToGeneral
| GeneralToSpecific
| SpecificToSpecific

-- Define a function that describes the direction of each reasoning type
def reasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.SpecificToGeneral
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating that the reasoning directions are correct
theorem correct_reasoning_directions :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.SpecificToGeneral) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end NUMINAMATH_CALUDE_correct_reasoning_directions_l477_47793


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_pi_3_l477_47773

theorem cos_2alpha_minus_pi_3 (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 4) :
  Real.cos (2 * α - π / 3) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_pi_3_l477_47773


namespace NUMINAMATH_CALUDE_childrens_tickets_l477_47721

theorem childrens_tickets (adult_price child_price total_tickets total_cost : ℚ) 
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_tickets child_tickets : ℚ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_cost ∧
    child_tickets = 16 := by
  sorry

end NUMINAMATH_CALUDE_childrens_tickets_l477_47721


namespace NUMINAMATH_CALUDE_fraction_meaningful_l477_47709

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (1 - x) / (1 + x)) ↔ x ≠ -1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l477_47709


namespace NUMINAMATH_CALUDE_class_average_mark_l477_47780

theorem class_average_mark (students1 students2 : ℕ) (avg2 avg_combined : ℚ) 
  (h1 : students1 = 30)
  (h2 : students2 = 50)
  (h3 : avg2 = 70)
  (h4 : avg_combined = 58.75)
  (h5 : (students1 : ℚ) * x + (students2 : ℚ) * avg2 = ((students1 : ℚ) + (students2 : ℚ)) * avg_combined) :
  x = 40 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l477_47780


namespace NUMINAMATH_CALUDE_solution_characterization_l477_47726

def SolutionSet : Set (ℕ × ℕ × ℕ) := {(1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 3, 2), (3, 5, 4), (2, 1, 1), (2, 1, 3), (4, 3, 5), (5, 4, 3), (3, 2, 1)}

def DivisibilityCondition (x y z : ℕ) : Prop :=
  (x ∣ y + 1) ∧ (y ∣ z + 1) ∧ (z ∣ x + 1)

theorem solution_characterization :
  ∀ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0) →
    (DivisibilityCondition x y z ↔ (x, y, z) ∈ SolutionSet) := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l477_47726


namespace NUMINAMATH_CALUDE_range_when_p_true_range_when_one_true_one_false_l477_47791

-- Define the propositions
def has_two_distinct_negative_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0

-- Theorem 1
theorem range_when_p_true (m : ℝ) :
  has_two_distinct_negative_roots m → m > 2 :=
sorry

-- Theorem 2
theorem range_when_one_true_one_false (m : ℝ) :
  (has_two_distinct_negative_roots m ↔ ¬has_no_real_roots m) →
  (m ∈ Set.Ioo 1 2 ∪ Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_range_when_p_true_range_when_one_true_one_false_l477_47791


namespace NUMINAMATH_CALUDE_gcd_372_684_l477_47722

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_372_684_l477_47722


namespace NUMINAMATH_CALUDE_ticket_cost_proof_l477_47732

def adult_price : ℕ := 12
def child_price : ℕ := 10
def senior_price : ℕ := 8
def student_price : ℕ := 9

def num_parents : ℕ := 2
def num_grandparents : ℕ := 2
def num_sisters : ℕ := 3
def num_cousins : ℕ := 1
def num_uncle_aunt : ℕ := 2

def total_cost : ℕ :=
  num_parents * adult_price +
  num_grandparents * senior_price +
  num_sisters * child_price +
  num_cousins * student_price +
  num_uncle_aunt * adult_price

theorem ticket_cost_proof : total_cost = 103 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_proof_l477_47732


namespace NUMINAMATH_CALUDE_solution_range_l477_47754

-- Define the system of inequalities
def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, a * x > -1 ∧ x + a > 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -1 ∨ a ≥ 0

-- Theorem statement
theorem solution_range :
  ∀ a : ℝ, has_solution a ↔ a_range a := by sorry

end NUMINAMATH_CALUDE_solution_range_l477_47754


namespace NUMINAMATH_CALUDE_exponent_relations_l477_47779

theorem exponent_relations (a : ℝ) (m n k : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : a^m = 2) 
  (h3 : a^n = 4) 
  (h4 : a^k = 32) : 
  (a^(3*m + 2*n - k) = 4) ∧ (k - 3*m - n = 0) := by
  sorry

end NUMINAMATH_CALUDE_exponent_relations_l477_47779


namespace NUMINAMATH_CALUDE_competition_results_l477_47702

def seventh_grade_scores : List ℕ := [3, 6, 7, 6, 6, 8, 6, 9, 6, 10]
def eighth_grade_scores : List ℕ := [5, 6, 8, 7, 5, 8, 7, 9, 8, 8]

def xiao_li_score : ℕ := 7
def xiao_zhang_score : ℕ := 7

def mode (l : List ℕ) : ℕ := sorry
def average (l : List ℕ) : ℚ := sorry
def median (l : List ℕ) : ℚ := sorry

theorem competition_results :
  (mode seventh_grade_scores = 6) ∧
  (average eighth_grade_scores = 7.1) ∧
  (median seventh_grade_scores = 6) ∧
  (median eighth_grade_scores = 7.5) ∧
  (xiao_li_score > median seventh_grade_scores) ∧
  (xiao_zhang_score < median eighth_grade_scores) := by
  sorry

#check competition_results

end NUMINAMATH_CALUDE_competition_results_l477_47702


namespace NUMINAMATH_CALUDE_no_integer_solutions_l477_47788

theorem no_integer_solutions : ¬∃ (x y z : ℤ),
  (x^2 - 4*x*y + 3*y^2 - z^2 = 25) ∧
  (-x^2 + 4*y*z + 3*z^2 = 36) ∧
  (x^2 + 2*x*y + 9*z^2 = 121) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l477_47788


namespace NUMINAMATH_CALUDE_jose_investment_is_45000_l477_47766

/-- Represents the investment scenario of Tom and Jose -/
structure InvestmentScenario where
  tom_investment : ℕ
  tom_months : ℕ
  jose_months : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given scenario -/
def calculate_jose_investment (scenario : InvestmentScenario) : ℕ :=
  (scenario.jose_profit * scenario.tom_investment * scenario.tom_months) /
  (scenario.tom_months * (scenario.total_profit - scenario.jose_profit))

/-- Theorem stating that Jose's investment is 45000 given the specific scenario -/
theorem jose_investment_is_45000 (scenario : InvestmentScenario)
  (h1 : scenario.tom_investment = 30000)
  (h2 : scenario.tom_months = 12)
  (h3 : scenario.jose_months = 10)
  (h4 : scenario.total_profit = 45000)
  (h5 : scenario.jose_profit = 25000) :
  calculate_jose_investment scenario = 45000 := by
  sorry


end NUMINAMATH_CALUDE_jose_investment_is_45000_l477_47766


namespace NUMINAMATH_CALUDE_range_equal_shifted_l477_47704

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the range of a function
def range (g : ℝ → ℝ) := {y : ℝ | ∃ x, g x = y}

-- Theorem statement
theorem range_equal_shifted : range f = range (fun x ↦ f (x + 1)) := by sorry

end NUMINAMATH_CALUDE_range_equal_shifted_l477_47704


namespace NUMINAMATH_CALUDE_remainder_divisibility_l477_47797

theorem remainder_divisibility (x : ℤ) : x % 52 = 19 → x % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l477_47797


namespace NUMINAMATH_CALUDE_rectangle_side_relation_l477_47796

/-- Given a rectangle with adjacent sides x and y, and area 30, prove that y = 30/x -/
theorem rectangle_side_relation (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 30) : 
  y = 30 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_relation_l477_47796


namespace NUMINAMATH_CALUDE_quadratic_set_equality_l477_47743

theorem quadratic_set_equality (p : ℝ) : 
  ({x : ℝ | x^2 - 5*x + p ≥ 0} = {x : ℝ | x ≤ -1 ∨ x ≥ 6}) → p = -6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_set_equality_l477_47743


namespace NUMINAMATH_CALUDE_sum_of_reversed_square_digits_l477_47785

/-- The number to be squared -/
def n : ℕ := 11111

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ := sorry

/-- Function to reverse the digits of a natural number -/
def reverse_digits (m : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the digits of the reversed square of 11111 is 25 -/
theorem sum_of_reversed_square_digits : sum_of_digits (reverse_digits (n^2)) = 25 := by sorry

end NUMINAMATH_CALUDE_sum_of_reversed_square_digits_l477_47785


namespace NUMINAMATH_CALUDE_abs_g_zero_eq_forty_l477_47748

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that the absolute value of g at specific points is 10 -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  |g (-2)| = 10 ∧ |g 1| = 10 ∧ |g 4| = 10 ∧ |g 5| = 10 ∧ |g 6| = 10 ∧ |g 9| = 10

/-- The theorem stating that if g satisfies the specific values, then |g(0)| = 40 -/
theorem abs_g_zero_eq_forty (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g 0| = 40 := by
  sorry

end NUMINAMATH_CALUDE_abs_g_zero_eq_forty_l477_47748


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_two_l477_47720

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 2 is -2 -/
theorem opposite_of_two : opposite 2 = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_two_l477_47720


namespace NUMINAMATH_CALUDE_multiply_by_15_subtract_1_l477_47758

theorem multiply_by_15_subtract_1 (x : ℝ) : 15 * x = 45 → x - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_15_subtract_1_l477_47758


namespace NUMINAMATH_CALUDE_circle_triangle_intersection_l477_47730

/-- Given an equilateral triangle intersected by a circle at six points, 
    this theorem proves the length of DE based on other given lengths. -/
theorem circle_triangle_intersection (AG GF FC HJ : ℝ) (h1 : AG = 2) (h2 : GF = 13) 
  (h3 : FC = 1) (h4 : HJ = 7) : ∃ (DE : ℝ), DE = 2 * Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_intersection_l477_47730


namespace NUMINAMATH_CALUDE_emma_remaining_time_l477_47795

-- Define the wrapping rates and initial joint work time
def emma_rate : ℚ := 1 / 6
def troy_rate : ℚ := 1 / 8
def joint_work_time : ℚ := 2

-- Define the function to calculate the remaining time for Emma
def remaining_time_for_emma (emma_rate troy_rate joint_work_time : ℚ) : ℚ :=
  let joint_completion := (emma_rate + troy_rate) * joint_work_time
  let remaining_work := 1 - joint_completion
  remaining_work / emma_rate

-- Theorem statement
theorem emma_remaining_time :
  remaining_time_for_emma emma_rate troy_rate joint_work_time = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_emma_remaining_time_l477_47795


namespace NUMINAMATH_CALUDE_sandra_feeding_days_l477_47782

/-- The number of days Sandra can feed the puppies with the given formula -/
def feeding_days (num_puppies : ℕ) (total_portions : ℕ) (feedings_per_day : ℕ) : ℕ :=
  total_portions / (num_puppies * feedings_per_day)

/-- Theorem stating that Sandra can feed the puppies for 5 days with the given formula -/
theorem sandra_feeding_days :
  feeding_days 7 105 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sandra_feeding_days_l477_47782


namespace NUMINAMATH_CALUDE_mirror_area_l477_47790

/-- Given a rectangular frame with outer dimensions 100 cm by 140 cm and a uniform frame width of 15 cm,
    the area of the rectangular mirror that fits exactly inside the frame is 7700 cm². -/
theorem mirror_area (frame_width : ℝ) (frame_height : ℝ) (frame_thickness : ℝ) : 
  frame_width = 100 ∧ frame_height = 140 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 7700 := by
  sorry

end NUMINAMATH_CALUDE_mirror_area_l477_47790


namespace NUMINAMATH_CALUDE_unique_solution_l477_47737

/-- The # operation as defined in the problem -/
def hash (a b : ℝ) : ℝ := a * b - 2 * a - 2 * b + 6

/-- Statement of the problem -/
theorem unique_solution : ∃! (x : ℝ), x > 0 ∧ hash (hash x 7) x = 82 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l477_47737


namespace NUMINAMATH_CALUDE_optimal_labeled_price_l477_47778

/-- Represents the pricing strategy of a retailer --/
structure RetailPricing where
  list_price : ℝ
  purchase_discount : ℝ
  sale_discount : ℝ
  profit_margin : ℝ
  labeled_price : ℝ

/-- The pricing strategy satisfies the retailer's conditions --/
def satisfies_conditions (rp : RetailPricing) : Prop :=
  rp.purchase_discount = 0.3 ∧
  rp.sale_discount = 0.25 ∧
  rp.profit_margin = 0.3 ∧
  rp.labeled_price > 0 ∧
  rp.list_price > 0

/-- The final selling price after discount --/
def selling_price (rp : RetailPricing) : ℝ :=
  rp.labeled_price * (1 - rp.sale_discount)

/-- The purchase price for the retailer --/
def purchase_price (rp : RetailPricing) : ℝ :=
  rp.list_price * (1 - rp.purchase_discount)

/-- The profit calculation --/
def profit (rp : RetailPricing) : ℝ :=
  selling_price rp - purchase_price rp

/-- The theorem stating that the labeled price should be 135% of the list price --/
theorem optimal_labeled_price (rp : RetailPricing) 
  (h : satisfies_conditions rp) : 
  rp.labeled_price = 1.35 * rp.list_price ↔ 
  profit rp = rp.profit_margin * selling_price rp :=
sorry

end NUMINAMATH_CALUDE_optimal_labeled_price_l477_47778


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l477_47714

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), (n > 0) ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0) → 
    m ≥ n) ∧
  n = 210 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l477_47714


namespace NUMINAMATH_CALUDE_negation_of_at_most_one_l477_47763

theorem negation_of_at_most_one (P : Type → Prop) :
  (¬ (∃! x, P x)) ↔ (∃ x y, P x ∧ P y ∧ x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_at_most_one_l477_47763


namespace NUMINAMATH_CALUDE_ratio_equality_l477_47756

theorem ratio_equality (a b c u v w : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_u : 0 < u) (h_pos_v : 0 < v) (h_pos_w : 0 < w)
  (h_sum_abc : a^2 + b^2 + c^2 = 9)
  (h_sum_uvw : u^2 + v^2 + w^2 = 49)
  (h_dot_product : a*u + b*v + c*w = 21) :
  (a + b + c) / (u + v + w) = 3/7 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l477_47756


namespace NUMINAMATH_CALUDE_combination_formula_l477_47701

/-- The number of combinations of n things taken k at a time -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem combination_formula (n m : ℕ) (h : n ≥ m - 1) :
  binomial n (m - 1) = Nat.factorial n / (Nat.factorial (m - 1) * Nat.factorial (n - m + 1)) := by
  sorry

end NUMINAMATH_CALUDE_combination_formula_l477_47701


namespace NUMINAMATH_CALUDE_prob_same_type_three_pairs_l477_47781

/-- Represents a collection of paired items -/
structure PairedCollection :=
  (num_pairs : ℕ)
  (items_per_pair : ℕ)
  (total_items : ℕ)
  (h_total : total_items = num_pairs * items_per_pair)

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the probability of selecting two items of the same type -/
def prob_same_type (collection : PairedCollection) : ℚ :=
  (collection.num_pairs : ℚ) / (choose collection.total_items 2)

/-- The main theorem to be proved -/
theorem prob_same_type_three_pairs :
  let shoe_collection : PairedCollection :=
    { num_pairs := 3
    , items_per_pair := 2
    , total_items := 6
    , h_total := rfl }
  prob_same_type shoe_collection = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_type_three_pairs_l477_47781


namespace NUMINAMATH_CALUDE_wood_rope_problem_l477_47711

/-- Represents the system of equations for the wood and rope problem -/
def wood_rope_equations (x y : ℝ) : Prop :=
  (y - x = 4.5) ∧ (x - y/2 = 1)

/-- Theorem stating that the equations correctly represent the given conditions -/
theorem wood_rope_problem (x y : ℝ) :
  wood_rope_equations x y →
  (y - x = 4.5 ∧ x - y/2 = 1) :=
by
  sorry

#check wood_rope_problem

end NUMINAMATH_CALUDE_wood_rope_problem_l477_47711


namespace NUMINAMATH_CALUDE_cafeteria_pies_l477_47712

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) 
  (h1 : initial_apples = 50)
  (h2 : handed_out = 5)
  (h3 : apples_per_pie = 5) :
  (initial_apples - handed_out) / apples_per_pie = 9 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l477_47712


namespace NUMINAMATH_CALUDE_triangle_height_theorem_l477_47755

-- Define the triangle ABC
theorem triangle_height_theorem (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  1 + 2 * Real.cos (B + C) = 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  -- Conclusion
  b * Real.sin C = (Real.sqrt 3 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_theorem_l477_47755


namespace NUMINAMATH_CALUDE_train_length_l477_47741

/-- The length of a train given its crossing time, bridge length, and speed. -/
theorem train_length (crossing_time : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) :
  crossing_time = 29.997600191984642 →
  bridge_length = 200 →
  train_speed_kmph = 36 →
  ∃ (train_length : ℝ), abs (train_length - 99.976) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l477_47741


namespace NUMINAMATH_CALUDE_ratio_bounds_in_acute_triangle_l477_47798

theorem ratio_bounds_in_acute_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  -- Sum of angles in a triangle
  A + B + C = π ∧
  -- A = 2B
  A = 2 * B ∧
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  c / (Real.sin C) = a / (Real.sin A) →
  -- Conclusion: a/b is bounded by √2 and √3
  Real.sqrt 2 < a / b ∧ a / b < Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ratio_bounds_in_acute_triangle_l477_47798


namespace NUMINAMATH_CALUDE_hoseok_calculation_l477_47771

theorem hoseok_calculation : ∃ x : ℤ, (x - 7 = 9) ∧ (3 * x = 48) := by
  sorry

end NUMINAMATH_CALUDE_hoseok_calculation_l477_47771


namespace NUMINAMATH_CALUDE_cubic_function_three_zeros_l477_47789

/-- A cubic function with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - k

/-- The derivative of f with respect to x -/
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem cubic_function_three_zeros (k : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f k x = 0 ∧ f k y = 0 ∧ f k z = 0) →
  -4 < k ∧ k < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_three_zeros_l477_47789


namespace NUMINAMATH_CALUDE_min_positive_temperatures_l477_47733

theorem min_positive_temperatures (x y : ℕ) : 
  x * (x - 1) = 90 →
  y * (y - 1) + (10 - y) * (9 - y) = 48 →
  y ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_positive_temperatures_l477_47733


namespace NUMINAMATH_CALUDE_yurts_are_xarps_and_zarqs_l477_47799

-- Define the sets
variable (U : Type) -- Universe set
variable (Xarp Zarq Yurt Wint : Set U)

-- Define the conditions
variable (h1 : Xarp ⊆ Zarq)
variable (h2 : Yurt ⊆ Zarq)
variable (h3 : Xarp ⊆ Wint)
variable (h4 : Yurt ⊆ Xarp)

-- Theorem to prove
theorem yurts_are_xarps_and_zarqs : Yurt ⊆ Xarp ∩ Zarq :=
sorry

end NUMINAMATH_CALUDE_yurts_are_xarps_and_zarqs_l477_47799


namespace NUMINAMATH_CALUDE_x_value_l477_47731

theorem x_value (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l477_47731


namespace NUMINAMATH_CALUDE_bottle_volume_is_one_and_half_quarts_l477_47703

/-- Represents the daily water consumption of Tim -/
structure DailyWaterConsumption where
  bottles : ℕ := 2
  additional_ounces : ℕ := 20

/-- Represents the weekly water consumption in ounces -/
def weekly_ounces : ℕ := 812

/-- Conversion factor from ounces to quarts -/
def ounces_per_quart : ℕ := 32

/-- Number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating that each bottle contains 1.5 quarts of water -/
theorem bottle_volume_is_one_and_half_quarts 
  (daily : DailyWaterConsumption) 
  (h1 : daily.bottles = 2) 
  (h2 : daily.additional_ounces = 20) :
  (weekly_ounces : ℚ) / (ounces_per_quart * days_per_week * daily.bottles : ℚ) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_bottle_volume_is_one_and_half_quarts_l477_47703


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l477_47700

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  parallel a b → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l477_47700
