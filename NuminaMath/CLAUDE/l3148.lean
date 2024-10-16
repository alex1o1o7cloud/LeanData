import Mathlib

namespace NUMINAMATH_CALUDE_total_paths_A_to_C_l3148_314815

/-- The number of paths between two points -/
def num_paths (start finish : ℕ) : ℕ := sorry

theorem total_paths_A_to_C : 
  let paths_A_to_B := num_paths 1 2
  let paths_B_to_D := num_paths 2 3
  let paths_D_to_C := num_paths 3 4
  let direct_paths_A_to_C := num_paths 1 4
  
  paths_A_to_B = 2 →
  paths_B_to_D = 2 →
  paths_D_to_C = 2 →
  direct_paths_A_to_C = 2 →
  
  paths_A_to_B * paths_B_to_D * paths_D_to_C + direct_paths_A_to_C = 10 :=
by sorry

end NUMINAMATH_CALUDE_total_paths_A_to_C_l3148_314815


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l3148_314872

theorem integer_pair_divisibility (m n : ℕ+) :
  (∃ k : ℤ, (m : ℤ) + n^2 = k * ((m : ℤ)^2 - n)) ∧
  (∃ l : ℤ, (m : ℤ)^2 + n = l * (n^2 - m)) →
  ((m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨
   (m = 2 ∧ n = 1) ∨ (m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l3148_314872


namespace NUMINAMATH_CALUDE_sandbox_length_l3148_314856

theorem sandbox_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 146 → area = 45552 → length * width = area → length = 312 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_length_l3148_314856


namespace NUMINAMATH_CALUDE_double_quarter_four_percent_l3148_314854

theorem double_quarter_four_percent : (2 * (1/4 * (4/100))) = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_double_quarter_four_percent_l3148_314854


namespace NUMINAMATH_CALUDE_smallest_among_four_l3148_314858

theorem smallest_among_four (a b c d : ℝ) :
  a = |-2| ∧ b = -1 ∧ c = 0 ∧ d = -1/2 →
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_four_l3148_314858


namespace NUMINAMATH_CALUDE_problem_solution_l3148_314862

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 3) (h2 : y^2 / z = 4) (h3 : z^2 / x = 5) :
  x = (36 * Real.sqrt 5) ^ (4/11) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3148_314862


namespace NUMINAMATH_CALUDE_gcd_2025_2070_l3148_314855

theorem gcd_2025_2070 : Nat.gcd 2025 2070 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2025_2070_l3148_314855


namespace NUMINAMATH_CALUDE_opposite_of_negative_abs_two_fifths_l3148_314892

theorem opposite_of_negative_abs_two_fifths :
  -(- |2 / 5|) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_abs_two_fifths_l3148_314892


namespace NUMINAMATH_CALUDE_sequence_congruence_l3148_314870

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ → ℤ
  | 0, _ => 0
  | 1, _ => 1
  | (n + 2), k => 2 * k * a (n + 1) k - (k^2 + 1) * a n k

/-- Main theorem -/
theorem sequence_congruence (k : ℤ) (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 4 = 3) :
  (∀ n : ℕ, a (n + p^2 - 1) k ≡ a n k [ZMOD p]) ∧
  (∀ n : ℕ, a (n + p^3 - p) k ≡ a n k [ZMOD p^2]) := by
  sorry

#check sequence_congruence

end NUMINAMATH_CALUDE_sequence_congruence_l3148_314870


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3148_314830

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + (a - 1) * x + 1 > 0) ↔ (1 ≤ a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3148_314830


namespace NUMINAMATH_CALUDE_find_unknown_areas_l3148_314882

/-- Represents the areas of rectangles in a divided larger rectangle -/
structure RectangleAreas where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  area5 : ℝ
  a : ℝ
  b : ℝ

/-- Theorem stating the values of unknown areas a and b given other known areas -/
theorem find_unknown_areas (areas : RectangleAreas) 
  (h1 : areas.area1 = 25)
  (h2 : areas.area2 = 104)
  (h3 : areas.area3 = 40)
  (h4 : areas.area4 = 143)
  (h5 : areas.area5 = 66)
  (h6 : areas.area2 / areas.area3 = areas.area4 / areas.b)
  (h7 : areas.area1 / areas.a = areas.b / areas.area5) :
  areas.a = 30 ∧ areas.b = 55 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_areas_l3148_314882


namespace NUMINAMATH_CALUDE_frog_to_hamster_ratio_l3148_314851

-- Define the lifespans of the animals
def bat_lifespan : ℕ := 10
def hamster_lifespan : ℕ := bat_lifespan - 6

-- Define the total lifespan
def total_lifespan : ℕ := 30

-- Define the frog's lifespan as a function of the hamster's
def frog_lifespan : ℕ := total_lifespan - (bat_lifespan + hamster_lifespan)

-- Theorem to prove
theorem frog_to_hamster_ratio :
  frog_lifespan / hamster_lifespan = 4 :=
by sorry

end NUMINAMATH_CALUDE_frog_to_hamster_ratio_l3148_314851


namespace NUMINAMATH_CALUDE_work_completion_proof_l3148_314809

/-- The number of days it takes for b to complete the work alone -/
def b_days : ℝ := 26.25

/-- The number of days it takes for a to complete the work alone -/
def a_days : ℝ := 24

/-- The number of days it takes for c to complete the work alone -/
def c_days : ℝ := 40

/-- The total number of days it took to complete the work -/
def total_days : ℝ := 11

/-- The number of days c worked before leaving -/
def c_work_days : ℝ := total_days - 4

theorem work_completion_proof :
  7 * (1 / a_days + 1 / b_days + 1 / c_days) + 4 * (1 / a_days + 1 / b_days) = 1 := by
  sorry

#check work_completion_proof

end NUMINAMATH_CALUDE_work_completion_proof_l3148_314809


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3148_314886

/-- An arithmetic sequence satisfying given conditions has one of two specific general terms -/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : a 2 + a 7 + a 12 = 12) 
  (h_product : a 2 * a 7 * a 12 = 28) :
  (∃ C : ℚ, ∀ n : ℕ, a n = 3/5 * n - 1/5 + C) ∨ 
  (∃ C : ℚ, ∀ n : ℕ, a n = -3/5 * n + 41/5 + C) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3148_314886


namespace NUMINAMATH_CALUDE_sticker_difference_l3148_314841

/-- Given two people with the same initial number of stickers, if one person uses 15 stickers
    and the other buys 18 stickers, the difference in their final number of stickers is 33. -/
theorem sticker_difference (initial_stickers : ℕ) : 
  (initial_stickers + 18) - (initial_stickers - 15) = 33 := by
  sorry

end NUMINAMATH_CALUDE_sticker_difference_l3148_314841


namespace NUMINAMATH_CALUDE_plane_distance_l3148_314897

/-- Proves that a plane flying east at 300 km/h and west at 400 km/h for a total of 7 hours travels 1200 km from the airport -/
theorem plane_distance (speed_east speed_west total_time : ℝ) 
  (h_speed_east : speed_east = 300)
  (h_speed_west : speed_west = 400)
  (h_total_time : total_time = 7) :
  ∃ (time_east time_west distance : ℝ),
    time_east + time_west = total_time ∧
    speed_east * time_east = distance ∧
    speed_west * time_west = distance ∧
    distance = 1200 := by
  sorry

end NUMINAMATH_CALUDE_plane_distance_l3148_314897


namespace NUMINAMATH_CALUDE_soccer_team_size_l3148_314842

/-- The number of players prepared for a soccer game -/
def players_prepared (starting_players : ℕ) (first_half_subs : ℕ) (second_half_subs : ℕ) (non_playing_players : ℕ) : ℕ :=
  starting_players + first_half_subs + non_playing_players

theorem soccer_team_size :
  let starting_players : ℕ := 11
  let first_half_subs : ℕ := 2
  let second_half_subs : ℕ := 2 * first_half_subs
  let non_playing_players : ℕ := 7
  players_prepared starting_players first_half_subs second_half_subs non_playing_players = 20 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_size_l3148_314842


namespace NUMINAMATH_CALUDE_parallelogram_reconstruction_l3148_314802

/-- Given a parallelogram ABCD with E as the midpoint of BC and F as the midpoint of CD,
    prove that the coordinates of C can be determined from the coordinates of A, E, and F. -/
theorem parallelogram_reconstruction (A E F : ℝ × ℝ) :
  let K : ℝ × ℝ := ((E.1 + F.1) / 2, (E.2 + F.2) / 2)
  let C : ℝ × ℝ := (A.1 / 2, A.2 / 2)
  (∃ (B D : ℝ × ℝ), 
    -- ABCD is a parallelogram
    (A.1 - B.1 = D.1 - C.1 ∧ A.2 - B.2 = D.2 - C.2) ∧
    (A.1 - D.1 = B.1 - C.1 ∧ A.2 - D.2 = B.2 - C.2) ∧
    -- E is the midpoint of BC
    (E.1 = (B.1 + C.1) / 2 ∧ E.2 = (B.2 + C.2) / 2) ∧
    -- F is the midpoint of CD
    (F.1 = (C.1 + D.1) / 2 ∧ F.2 = (C.2 + D.2) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_reconstruction_l3148_314802


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3148_314898

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^4 / (a^3 + a^2*b + a*b^2 + b^3) +
   b^4 / (b^3 + b^2*c + b*c^2 + c^3) +
   c^4 / (c^3 + c^2*d + c*d^2 + d^3) +
   d^4 / (d^3 + d^2*a + d*a^2 + a^3)) ≥ (a + b + c + d) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3148_314898


namespace NUMINAMATH_CALUDE_tourists_travelers_checks_l3148_314871

/-- Represents the number of travelers checks of each denomination -/
structure TravelersChecks where
  fifty : Nat
  hundred : Nat

/-- The problem statement -/
theorem tourists_travelers_checks 
  (tc : TravelersChecks)
  (h1 : 50 * tc.fifty + 100 * tc.hundred = 1800)
  (h2 : tc.fifty ≥ 24)
  (h3 : (1800 - 50 * 24) / (tc.fifty + tc.hundred - 24) = 100) :
  tc.fifty + tc.hundred = 30 := by
  sorry

end NUMINAMATH_CALUDE_tourists_travelers_checks_l3148_314871


namespace NUMINAMATH_CALUDE_ellipse_axes_sum_l3148_314889

-- Define the cylinder and spheres
def cylinder_radius : ℝ := 6
def sphere_radius : ℝ := 6
def sphere_centers_distance : ℝ := 13

-- Define the ellipse axes
def minor_axis : ℝ := 2 * cylinder_radius
def major_axis : ℝ := sphere_centers_distance

-- Theorem statement
theorem ellipse_axes_sum :
  minor_axis + major_axis = 25 := by sorry

end NUMINAMATH_CALUDE_ellipse_axes_sum_l3148_314889


namespace NUMINAMATH_CALUDE_correct_calculation_l3148_314825

theorem correct_calculation (x : ℝ) (h : 15 * x = 45) : 5 * x = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3148_314825


namespace NUMINAMATH_CALUDE_fourth_day_temperature_l3148_314869

/-- Given three temperatures and a four-day average, calculate the fourth temperature --/
theorem fourth_day_temperature 
  (temp1 temp2 temp3 : ℤ) 
  (average : ℚ) 
  (h1 : temp1 = -36)
  (h2 : temp2 = -15)
  (h3 : temp3 = -10)
  (h4 : average = -12)
  : (4 : ℚ) * average - (temp1 + temp2 + temp3 : ℚ) = 13 := by
  sorry

#check fourth_day_temperature

end NUMINAMATH_CALUDE_fourth_day_temperature_l3148_314869


namespace NUMINAMATH_CALUDE_multiple_count_l3148_314891

theorem multiple_count (n : ℕ) (h1 : n > 0) (h2 : n ≤ 400) : 
  (∃ (k : ℕ), k > 0 ∧ (∀ m : ℕ, m > 0 → m ≤ 400 → m % k = 0 → m ∈ Finset.range 401) ∧ 
  (Finset.filter (λ m => m % k = 0) (Finset.range 401)).card = 16) → 
  n = 25 :=
sorry

end NUMINAMATH_CALUDE_multiple_count_l3148_314891


namespace NUMINAMATH_CALUDE_enclosed_area_circular_arcs_octagon_l3148_314803

/-- The area enclosed by a curve formed by circular arcs centered on a regular octagon -/
theorem enclosed_area_circular_arcs_octagon (n : ℕ) (arc_length : ℝ) (side_length : ℝ) : 
  n = 12 → 
  arc_length = 3 * π / 4 → 
  side_length = 3 → 
  ∃ (area : ℝ), area = 54 + 18 * Real.sqrt 2 + 81 * π / 64 - 54 * π / 64 - 18 * π * Real.sqrt 2 / 64 :=
by sorry

end NUMINAMATH_CALUDE_enclosed_area_circular_arcs_octagon_l3148_314803


namespace NUMINAMATH_CALUDE_square_rectangle_area_l3148_314810

/-- A rectangle composed of four identical squares with a given perimeter --/
structure SquareRectangle where
  side : ℝ  -- Side length of each square
  perim : ℝ  -- Perimeter of the rectangle
  perim_eq : perim = 10 * side  -- Perimeter equation

/-- The area of a SquareRectangle --/
def SquareRectangle.area (r : SquareRectangle) : ℝ := 4 * r.side^2

/-- Theorem: A SquareRectangle with perimeter 160 has an area of 1024 --/
theorem square_rectangle_area (r : SquareRectangle) (h : r.perim = 160) : r.area = 1024 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_l3148_314810


namespace NUMINAMATH_CALUDE_star_equation_solution_l3148_314828

/-- Custom binary operation -/
def star (a b : ℝ) : ℝ := a * b + a - 2 * b

/-- Theorem stating that if 3 star m = 17, then m = 14 -/
theorem star_equation_solution :
  ∀ m : ℝ, star 3 m = 17 → m = 14 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l3148_314828


namespace NUMINAMATH_CALUDE_monomial_equality_l3148_314873

-- Define variables
variable (a b : ℝ)
variable (x : ℝ)

-- Define the theorem
theorem monomial_equality (h : x * (2 * a^2 * b) = 2 * a^3 * b) : x = a := by
  sorry

end NUMINAMATH_CALUDE_monomial_equality_l3148_314873


namespace NUMINAMATH_CALUDE_smallest_value_theorem_l3148_314867

theorem smallest_value_theorem (v w : ℝ) 
  (h : ∀ (a b : ℝ), (2^(a+b) + 8)*(3^a + 3^b) ≤ v*(12^(a-1) + 12^(b-1) - 2^(a+b-1)) + w) : 
  (∀ (v' w' : ℝ), (∀ (a b : ℝ), (2^(a+b) + 8)*(3^a + 3^b) ≤ v'*(12^(a-1) + 12^(b-1) - 2^(a+b-1)) + w') → 
    128*v^2 + w^2 ≤ 128*v'^2 + w'^2) ∧ 128*v^2 + w^2 = 62208 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_theorem_l3148_314867


namespace NUMINAMATH_CALUDE_right_triangle_trig_sum_l3148_314837

theorem right_triangle_trig_sum (A B C : Real) : 
  -- Conditions
  A = π / 2 →  -- A = 90° in radians
  0 < B → B < π / 2 →  -- B is acute angle
  C = π / 2 - B →  -- C is complementary to B in right triangle
  -- Theorem
  Real.sin A + Real.sin B ^ 2 + Real.sin C ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_sum_l3148_314837


namespace NUMINAMATH_CALUDE_cos_pi_third_plus_two_alpha_l3148_314846

theorem cos_pi_third_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 3 - α) = 1 / 4) : 
  Real.cos (π / 3 + 2 * α) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_plus_two_alpha_l3148_314846


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3148_314814

def M : ℕ := 35 * 36 * 65 * 280

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry
def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_of_odd_divisors M) * 62 = sum_of_even_divisors M := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3148_314814


namespace NUMINAMATH_CALUDE_competition_result_l3148_314895

-- Define the participants
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

-- Define the places
inductive Place
| First
| Second
| Third
| Fourth

def is_odd_place (p : Place) : Bool :=
  match p with
  | Place.First => true
  | Place.Third => true
  | _ => false

def is_consecutive (p1 p2 : Place) : Bool :=
  match p1, p2 with
  | Place.First, Place.Second => true
  | Place.Second, Place.Third => true
  | Place.Third, Place.Fourth => true
  | Place.Second, Place.First => true
  | Place.Third, Place.Second => true
  | Place.Fourth, Place.Third => true
  | _, _ => false

def starts_with_O (p : Participant) : Bool :=
  match p with
  | Participant.Olya => true
  | Participant.Oleg => true
  | _ => false

def is_boy (p : Participant) : Bool :=
  match p with
  | Participant.Oleg => true
  | Participant.Pasha => true
  | _ => false

-- The main theorem
theorem competition_result :
  ∃! (result : Participant → Place),
    (∀ p, ∃! place, result p = place) ∧
    (∀ place, ∃! p, result p = place) ∧
    (∃! p, (result p = Place.First ∧ starts_with_O p) ∨
           (result p = Place.Second ∧ ¬starts_with_O p) ∨
           (result p = Place.Third ∧ ¬starts_with_O p) ∨
           (result p = Place.Fourth ∧ ¬starts_with_O p)) ∧
    (result Participant.Oleg = Place.First ∧
     result Participant.Olya = Place.Second ∧
     result Participant.Polya = Place.Third ∧
     result Participant.Pasha = Place.Fourth) :=
by sorry


end NUMINAMATH_CALUDE_competition_result_l3148_314895


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l3148_314868

def f (x : ℝ) := -x^2 + 1

theorem f_is_even_and_decreasing :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 0 < x ∧ x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l3148_314868


namespace NUMINAMATH_CALUDE_complex_fraction_magnitude_l3148_314894

theorem complex_fraction_magnitude : Complex.abs ((5 + Complex.I) / (1 - Complex.I)) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_magnitude_l3148_314894


namespace NUMINAMATH_CALUDE_always_bal_answer_l3148_314834

/-- Represents a guest in the castle -/
structure Guest where
  is_reliable : Bool

/-- Represents the possible questions that can be asked -/
inductive Question
  | q1  -- "Правильно ли ответить «бaл» на вопрос, надежны ли вы?"
  | q2  -- "Надежны ли вы в том и только в том случае, если «бaл» означает «да»?"

/-- The answer "бaл" -/
def bal : String := "бaл"

/-- Function representing a guest's response to a question -/
def guest_response (g : Guest) (q : Question) : String :=
  match q with
  | Question.q1 => bal
  | Question.q2 => bal

/-- Theorem stating that any guest will always answer "бaл" to either question -/
theorem always_bal_answer (g : Guest) (q : Question) :
  guest_response g q = bal := by sorry

end NUMINAMATH_CALUDE_always_bal_answer_l3148_314834


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_ratio_l3148_314844

/-- Given an ellipse and a hyperbola with common foci, prove that the ratio of their minor axes is √3 -/
theorem ellipse_hyperbola_ratio (a₁ b₁ a₂ b₂ c : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a₁ > b₁ ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 →
  P.1^2 / a₁^2 + P.2^2 / b₁^2 = 1 →
  P.1^2 / a₂^2 - P.2^2 / b₂^2 = 1 →
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 4 * c^2 →
  a₁^2 - b₁^2 = c^2 →
  a₂^2 + b₂^2 = c^2 →
  Real.cos (Real.arccos ((((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2).sqrt + ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt)^2 - 4*c^2) /
    (2 * ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2).sqrt * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt)) = 1/2 →
  b₁ / b₂ = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_ratio_l3148_314844


namespace NUMINAMATH_CALUDE_data_set_average_l3148_314808

theorem data_set_average (a : ℝ) : 
  (2 + 3 + 3 + 4 + a) / 5 = 3 → a = 3 := by
sorry

end NUMINAMATH_CALUDE_data_set_average_l3148_314808


namespace NUMINAMATH_CALUDE_line_parametric_equation_l3148_314866

theorem line_parametric_equation :
  ∀ (t : ℝ), 2 * (1 - t) - (3 - 2 * t) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_parametric_equation_l3148_314866


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l3148_314861

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l3148_314861


namespace NUMINAMATH_CALUDE_water_purification_equation_l3148_314890

/-- Represents the water purification scenario -/
structure WaterPurification where
  total_area : ℝ
  efficiency_increase : ℝ
  days_saved : ℝ
  daily_rate : ℝ

/-- Theorem stating the correct equation for the water purification scenario -/
theorem water_purification_equation (wp : WaterPurification) 
  (h1 : wp.total_area = 2400)
  (h2 : wp.efficiency_increase = 0.2)
  (h3 : wp.days_saved = 40)
  (h4 : wp.daily_rate > 0) :
  (wp.total_area * (1 + wp.efficiency_increase)) / wp.daily_rate - wp.total_area / wp.daily_rate = wp.days_saved :=
by sorry

end NUMINAMATH_CALUDE_water_purification_equation_l3148_314890


namespace NUMINAMATH_CALUDE_sum_of_sqrt_sequence_l3148_314835

theorem sum_of_sqrt_sequence :
  Real.sqrt 6 + Real.sqrt (6 + 8) + Real.sqrt (6 + 8 + 10) + 
  Real.sqrt (6 + 8 + 10 + 12) + Real.sqrt (6 + 8 + 10 + 12 + 14) = 
  Real.sqrt 6 + Real.sqrt 14 + Real.sqrt 24 + 6 + 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_sequence_l3148_314835


namespace NUMINAMATH_CALUDE_correct_calculation_l3148_314847

theorem correct_calculation (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3148_314847


namespace NUMINAMATH_CALUDE_no_solution_for_sock_problem_l3148_314831

theorem no_solution_for_sock_problem : ¬∃ (n m : ℕ), 
  n + m = 2009 ∧ 
  (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1) : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_sock_problem_l3148_314831


namespace NUMINAMATH_CALUDE_model_a_better_fit_l3148_314843

def model_a (x : ℝ) : ℝ := x^2 + 1
def model_b (x : ℝ) : ℝ := 3*x - 1

def data_points : List (ℝ × ℝ) := [(1, 2), (2, 5), (3, 10.2)]

def error (model : ℝ → ℝ) (point : ℝ × ℝ) : ℝ :=
  (model point.1 - point.2)^2

def total_error (model : ℝ → ℝ) (points : List (ℝ × ℝ)) : ℝ :=
  points.foldl (λ acc p => acc + error model p) 0

theorem model_a_better_fit :
  total_error model_a data_points < total_error model_b data_points := by
  sorry

end NUMINAMATH_CALUDE_model_a_better_fit_l3148_314843


namespace NUMINAMATH_CALUDE_triangle_ratio_sine_relation_l3148_314857

theorem triangle_ratio_sine_relation (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (b + c) / (c + a) = 4 / 5 ∧ (c + a) / (a + b) = 5 / 6 →
  (Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) + 
   Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) /
  Real.sin (Real.arccos ((c^2 + a^2 - b^2) / (2 * c * a))) = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_sine_relation_l3148_314857


namespace NUMINAMATH_CALUDE_soda_distribution_impossibility_l3148_314826

theorem soda_distribution_impossibility (total_sodas : ℕ) (sisters : ℕ) : 
  total_sodas = 9 →
  sisters = 2 →
  ¬∃ (sodas_per_sibling : ℕ), 
    sodas_per_sibling > 0 ∧ 
    total_sodas = sodas_per_sibling * (sisters + 2 * sisters) :=
by
  sorry

end NUMINAMATH_CALUDE_soda_distribution_impossibility_l3148_314826


namespace NUMINAMATH_CALUDE_number_of_divisors_l3148_314819

-- Define the number we're working with
def n : ℕ := 3465

-- Define the prime factorization of n
axiom prime_factorization : n = 3^2 * 5^1 * 7^2

-- Define the function to count positive divisors
def count_divisors (m : ℕ) : ℕ := sorry

-- Theorem stating the number of positive divisors of n
theorem number_of_divisors : count_divisors n = 18 := by sorry

end NUMINAMATH_CALUDE_number_of_divisors_l3148_314819


namespace NUMINAMATH_CALUDE_number_exceeds_fraction_l3148_314832

theorem number_exceeds_fraction : ∃ x : ℚ, x = (3/8) * x + 30 ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_fraction_l3148_314832


namespace NUMINAMATH_CALUDE_language_course_enrollment_l3148_314801

theorem language_course_enrollment (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_german : ℕ) (french_spanish : ℕ) (german_spanish : ℕ) (all_three : ℕ) :
  total = 120 →
  french = 52 →
  german = 35 →
  spanish = 48 →
  french_german = 15 →
  french_spanish = 20 →
  german_spanish = 12 →
  all_three = 6 →
  total - (french + german + spanish - french_german - french_spanish - german_spanish + all_three) = 32 := by
sorry

end NUMINAMATH_CALUDE_language_course_enrollment_l3148_314801


namespace NUMINAMATH_CALUDE_box_with_balls_l3148_314881

theorem box_with_balls (total : ℕ) (white blue red : ℕ) : 
  total = 100 →
  blue = white + 12 →
  red = 2 * blue →
  total = white + blue + red →
  white = 16 := by
sorry

end NUMINAMATH_CALUDE_box_with_balls_l3148_314881


namespace NUMINAMATH_CALUDE_logarithm_relation_l3148_314853

theorem logarithm_relation (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (ha1 : a ≠ 1) (hb1 : b ≠ 1) (hdist : a ≠ b ∧ a ≠ x ∧ b ≠ x)
  (heq : 4 * (Real.log x / Real.log a)^3 + 5 * (Real.log x / Real.log b)^3 = 7 * (Real.log x)^3) :
  ∃ k, b = a^k ∧ k = (3/5)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_relation_l3148_314853


namespace NUMINAMATH_CALUDE_rectangular_park_area_l3148_314820

/-- A rectangular park with a perimeter of 80 feet and length three times its width has an area of 300 square feet. -/
theorem rectangular_park_area : ∀ l w : ℝ,
  l > 0 → w > 0 →  -- Ensure positive dimensions
  2 * (l + w) = 80 →  -- Perimeter condition
  l = 3 * w →  -- Length is three times the width
  l * w = 300 := by
sorry

end NUMINAMATH_CALUDE_rectangular_park_area_l3148_314820


namespace NUMINAMATH_CALUDE_complex_number_location_l3148_314852

theorem complex_number_location : ∃ (z : ℂ), 
  z = (1 : ℂ) / (2 + Complex.I) + Complex.I ^ 2018 ∧ 
  z.re < 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3148_314852


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_l3148_314836

theorem triangle_inequality_expression (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : 
  (a - b)^2 - c^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_l3148_314836


namespace NUMINAMATH_CALUDE_squared_sum_equals_cube_root_l3148_314878

theorem squared_sum_equals_cube_root (x y : ℝ) 
  (h1 : x^2 - 3*y^2 = 17/x) 
  (h2 : 3*x^2 - y^2 = 23/y) : 
  x^2 + y^2 = 818^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_squared_sum_equals_cube_root_l3148_314878


namespace NUMINAMATH_CALUDE_calculation_proof_l3148_314807

theorem calculation_proof : 3 * (-4) - ((5 * (-5)) * (-2)) + 6 = -56 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3148_314807


namespace NUMINAMATH_CALUDE_max_value_ab_l3148_314863

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * y ≤ (1/4 : ℝ)) → a * b ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_value_ab_l3148_314863


namespace NUMINAMATH_CALUDE_wage_payment_days_l3148_314849

theorem wage_payment_days (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ∃ (total : ℝ), total > 0 ∧ total = 21 * a ∧ total = 28 * b → 
  ∃ (d : ℝ), d = 12 ∧ total = d * (a + b) := by
sorry

end NUMINAMATH_CALUDE_wage_payment_days_l3148_314849


namespace NUMINAMATH_CALUDE_maddie_weekend_watch_l3148_314845

def total_episodes : ℕ := 8
def episode_length : ℕ := 44
def monday_watch : ℕ := 138
def thursday_watch : ℕ := 21
def friday_episodes : ℕ := 2

def weekend_watch : ℕ := 105

theorem maddie_weekend_watch :
  let total_watch := total_episodes * episode_length
  let weekday_watch := monday_watch + thursday_watch + (friday_episodes * episode_length)
  total_watch - weekday_watch = weekend_watch := by sorry

end NUMINAMATH_CALUDE_maddie_weekend_watch_l3148_314845


namespace NUMINAMATH_CALUDE_appetizer_price_l3148_314874

theorem appetizer_price (total_spent : ℝ) (entree_percentage : ℝ) (num_appetizers : ℕ) 
  (h_total : total_spent = 50)
  (h_entree : entree_percentage = 0.8)
  (h_appetizers : num_appetizers = 2) : 
  (1 - entree_percentage) * total_spent / num_appetizers = 5 := by
  sorry

end NUMINAMATH_CALUDE_appetizer_price_l3148_314874


namespace NUMINAMATH_CALUDE_digit_sum_l3148_314813

/-- Given two digits x and y, if 3x * y4 = 156, then x + y = 13 -/
theorem digit_sum (x y : Nat) : 
  x ≤ 9 → y ≤ 9 → (30 + x) * (10 * y + 4) = 156 → x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_l3148_314813


namespace NUMINAMATH_CALUDE_nancy_total_games_l3148_314817

/-- The total number of games Nancy will attend over three months -/
def total_games (this_month : ℕ) (last_month : ℕ) (next_month : ℕ) : ℕ :=
  this_month + last_month + next_month

/-- Proof that Nancy will attend 24 games in total -/
theorem nancy_total_games :
  total_games 9 8 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_nancy_total_games_l3148_314817


namespace NUMINAMATH_CALUDE_no_valid_rectangle_l3148_314877

theorem no_valid_rectangle (a b x y : ℝ) : 
  a < b →
  x < a →
  y < a →
  2 * (x + y) = (2 * (a + b)) / 3 →
  x * y = (a * b) / 3 →
  False :=
by sorry

end NUMINAMATH_CALUDE_no_valid_rectangle_l3148_314877


namespace NUMINAMATH_CALUDE_vanessa_saves_three_weeks_l3148_314829

/-- Calculates the number of weeks needed to save for a dress -/
def weeks_to_save (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) : ℕ :=
  let additional_needed := dress_cost - initial_savings
  let net_weekly_savings := weekly_allowance - weekly_spending
  (additional_needed + net_weekly_savings - 1) / net_weekly_savings

/-- Proves that Vanessa needs 3 weeks to save for the dress -/
theorem vanessa_saves_three_weeks :
  weeks_to_save 80 20 30 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_saves_three_weeks_l3148_314829


namespace NUMINAMATH_CALUDE_ratio_problem_l3148_314879

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 2 / 7)
  (h3 : c / d = 4) :
  d / a = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3148_314879


namespace NUMINAMATH_CALUDE_arccos_sum_equation_l3148_314800

theorem arccos_sum_equation (x : ℝ) : 
  Real.arccos (3 * x) + Real.arccos x = π / 2 → x = 1 / Real.sqrt 10 ∨ x = -1 / Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sum_equation_l3148_314800


namespace NUMINAMATH_CALUDE_unique_cube_number_l3148_314896

theorem unique_cube_number : ∃! y : ℕ, 
  (∃ n : ℕ, y = n^3) ∧ 
  (y % 6 = 0) ∧ 
  (50 < y) ∧ 
  (y < 350) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_cube_number_l3148_314896


namespace NUMINAMATH_CALUDE_parabolas_similar_l3148_314806

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := x^2
def parabola2 (x : ℝ) : ℝ := 2 * x^2

-- Define a homothety transformation
def homothety (scale : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (scale * p.1, scale * p.2)

-- Theorem statement
theorem parabolas_similar :
  ∀ x : ℝ, homothety 2 (x, parabola2 x) = (2*x, parabola1 (2*x)) :=
by sorry

end NUMINAMATH_CALUDE_parabolas_similar_l3148_314806


namespace NUMINAMATH_CALUDE_feed_animals_count_l3148_314887

/-- Represents the number of pairs of animals in the zoo -/
def num_pairs : ℕ := 5

/-- Calculates the number of ways to feed all animals in the zoo -/
def feed_animals : ℕ :=
  (num_pairs) *  -- Choose from 5 females
  (num_pairs - 1) * (num_pairs - 1) *  -- Choose from 4 males, then 4 females
  (num_pairs - 2) * (num_pairs - 2) *  -- Choose from 3 males, then 3 females
  (num_pairs - 3) * (num_pairs - 3) *  -- Choose from 2 males, then 2 females
  (num_pairs - 4) * (num_pairs - 4)    -- Choose from 1 male, then 1 female

/-- Theorem stating the number of ways to feed all animals -/
theorem feed_animals_count : feed_animals = 2880 := by
  sorry

end NUMINAMATH_CALUDE_feed_animals_count_l3148_314887


namespace NUMINAMATH_CALUDE_all_lines_pass_through_fixed_point_l3148_314833

/-- A line in the xy-plane defined by the equation kx - y + 1 = k, where k is a real number. -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | k * p.1 - p.2 + 1 = k}

/-- The fixed point (1, 1) -/
def fixed_point : ℝ × ℝ := (1, 1)

/-- Theorem stating that all lines pass through the fixed point (1, 1) -/
theorem all_lines_pass_through_fixed_point :
  ∀ k : ℝ, fixed_point ∈ line k := by
  sorry


end NUMINAMATH_CALUDE_all_lines_pass_through_fixed_point_l3148_314833


namespace NUMINAMATH_CALUDE_flowers_per_set_l3148_314860

theorem flowers_per_set (total_flowers : ℕ) (num_sets : ℕ) (h1 : total_flowers = 270) (h2 : num_sets = 3) :
  total_flowers / num_sets = 90 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_set_l3148_314860


namespace NUMINAMATH_CALUDE_fertilizer_per_acre_l3148_314827

/-- Fertilizer problem --/
theorem fertilizer_per_acre 
  (horses : ℕ) 
  (fertilizer_per_horse_per_day : ℕ) 
  (acres : ℕ) 
  (acres_per_day : ℕ) 
  (total_days : ℕ) 
  (h1 : horses = 80)
  (h2 : fertilizer_per_horse_per_day = 5)
  (h3 : acres = 20)
  (h4 : acres_per_day = 4)
  (h5 : total_days = 25) :
  (horses * fertilizer_per_horse_per_day * total_days) / acres = 500 := by
  sorry

#check fertilizer_per_acre

end NUMINAMATH_CALUDE_fertilizer_per_acre_l3148_314827


namespace NUMINAMATH_CALUDE_fraction_sum_equals_two_l3148_314823

theorem fraction_sum_equals_two (a b : ℝ) (ha : a ≠ 0) : 
  (2*b + a) / a + (a - 2*b) / a = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_two_l3148_314823


namespace NUMINAMATH_CALUDE_square_difference_l3148_314821

theorem square_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 2) : x^2 - y^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3148_314821


namespace NUMINAMATH_CALUDE_max_garden_area_l3148_314816

/-- Given 420 feet of fencing to enclose a rectangular garden on three sides
    (with the fourth side against a wall), the maximum area that can be achieved
    is 22050 square feet. -/
theorem max_garden_area (fencing : ℝ) (h : fencing = 420) :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * l + w = fencing ∧
  ∀ (l' w' : ℝ), l' > 0 → w' > 0 → 2 * l' + w' = fencing →
  l * w ≥ l' * w' ∧ l * w = 22050 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l3148_314816


namespace NUMINAMATH_CALUDE_jury_duty_duration_l3148_314865

/-- Calculates the total number of days spent on jury duty -/
def total_jury_duty_days (jury_selection_days : ℕ) (trial_duration_factor : ℕ) 
  (deliberation_full_days : ℕ) (daily_deliberation_hours : ℕ) : ℕ :=
  let trial_days := jury_selection_days * trial_duration_factor
  let deliberation_hours := deliberation_full_days * 24
  let deliberation_days := deliberation_hours / daily_deliberation_hours
  jury_selection_days + trial_days + deliberation_days

/-- Theorem stating that the total number of days spent on jury duty is 19 -/
theorem jury_duty_duration : 
  total_jury_duty_days 2 4 6 16 = 19 := by
  sorry

end NUMINAMATH_CALUDE_jury_duty_duration_l3148_314865


namespace NUMINAMATH_CALUDE_john_nails_count_l3148_314838

/-- Calculates the total number of nails used in John's house wall construction --/
def total_nails (nails_per_plank : ℕ) (additional_nails : ℕ) (num_planks : ℕ) : ℕ :=
  nails_per_plank * num_planks + additional_nails

/-- Proves that John used 11 nails in total for his house wall construction --/
theorem john_nails_count :
  let nails_per_plank : ℕ := 3
  let additional_nails : ℕ := 8
  let num_planks : ℕ := 1
  total_nails nails_per_plank additional_nails num_planks = 11 := by
  sorry

end NUMINAMATH_CALUDE_john_nails_count_l3148_314838


namespace NUMINAMATH_CALUDE_sammy_has_twenty_caps_l3148_314888

/-- Represents the number of bottle caps each person has -/
structure BottleCaps where
  sammy : ℕ
  janine : ℕ
  billie : ℕ
  tommy : ℕ

/-- The initial state of bottle caps -/
def initial_state (b : ℕ) : BottleCaps :=
  { sammy := 3 * b + 2
    janine := 3 * b
    billie := b
    tommy := 0 }

/-- The final state of bottle caps after Billie's gift -/
def final_state (b : ℕ) : BottleCaps :=
  { sammy := 3 * b + 2
    janine := 3 * b
    billie := b - 4
    tommy := 4 }

/-- The theorem stating Sammy has 20 bottle caps -/
theorem sammy_has_twenty_caps :
  ∃ b : ℕ,
    (final_state b).tommy = 2 * (final_state b).billie ∧
    (final_state b).sammy = 20 := by
  sorry

#check sammy_has_twenty_caps

end NUMINAMATH_CALUDE_sammy_has_twenty_caps_l3148_314888


namespace NUMINAMATH_CALUDE_intersection_value_l3148_314876

theorem intersection_value (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (3 / a = b) ∧ (a - 1 = b) → 1 / a - 1 / b = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l3148_314876


namespace NUMINAMATH_CALUDE_power_of_two_triplets_l3148_314884

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def valid_triplet (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 2), (3, 2, 2), (2, 6, 11), (3, 5, 7),
   (2, 2, 2), (2, 3, 2), (6, 2, 11), (5, 3, 7),
   (2, 2, 2), (2, 2, 3), (11, 6, 2), (7, 5, 3)}

theorem power_of_two_triplets :
  ∀ a b c : ℕ, valid_triplet a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_power_of_two_triplets_l3148_314884


namespace NUMINAMATH_CALUDE_remaining_payment_prove_remaining_payment_l3148_314804

/-- Given a product with a deposit, sales tax, and processing fee, calculate the remaining amount to be paid -/
theorem remaining_payment (deposit_percentage : ℝ) (deposit_amount : ℝ) (sales_tax_percentage : ℝ) (processing_fee : ℝ) : ℝ :=
  let full_price := deposit_amount / deposit_percentage
  let sales_tax := sales_tax_percentage * full_price
  let total_additional_expenses := sales_tax + processing_fee
  full_price - deposit_amount + total_additional_expenses

/-- Prove that the remaining payment for the given conditions is $1520 -/
theorem prove_remaining_payment :
  remaining_payment 0.1 140 0.15 50 = 1520 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_prove_remaining_payment_l3148_314804


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l3148_314893

theorem complex_modulus_equation (n : ℝ) : 
  Complex.abs (6 + n * Complex.I) = 6 * Real.sqrt 5 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l3148_314893


namespace NUMINAMATH_CALUDE_line_points_determine_k_l3148_314859

/-- A line contains the points (6,10), (-2,k), and (-10,6). -/
def line_contains_points (k : ℝ) : Prop :=
  ∃ (m b : ℝ), 
    (10 = m * 6 + b) ∧
    (k = m * (-2) + b) ∧
    (6 = m * (-10) + b)

/-- If a line contains the points (6,10), (-2,k), and (-10,6), then k = 8. -/
theorem line_points_determine_k :
  ∀ k : ℝ, line_contains_points k → k = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_line_points_determine_k_l3148_314859


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3148_314811

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem parallel_line_through_point 
  (P : Point)
  (L1 : Line)
  (L2 : Line)
  (h1 : P.x = -1 ∧ P.y = 2)
  (h2 : L1.a = 2 ∧ L1.b = 1 ∧ L1.c = -5)
  (h3 : L2.a = 2 ∧ L2.b = 1 ∧ L2.c = 0)
  : parallel L1 L2 ∧ pointOnLine P L2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3148_314811


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3148_314875

/-- A rectangle with length thrice its breadth and area 108 square meters has a perimeter of 48 meters -/
theorem rectangle_perimeter (b : ℝ) (h1 : b > 0) (h2 : 3 * b * b = 108) : 2 * (3 * b + b) = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3148_314875


namespace NUMINAMATH_CALUDE_power_equation_solution_l3148_314885

theorem power_equation_solution : ∃ x : ℕ, 
  8 * (32 ^ 10) = 2 ^ x ∧ x = 53 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3148_314885


namespace NUMINAMATH_CALUDE_domino_distribution_l3148_314883

theorem domino_distribution (total_dominoes : Nat) (num_players : Nat) 
  (h1 : total_dominoes = 28) (h2 : num_players = 4) :
  total_dominoes / num_players = 7 := by
  sorry

end NUMINAMATH_CALUDE_domino_distribution_l3148_314883


namespace NUMINAMATH_CALUDE_train_speed_l3148_314848

/-- Calculates the speed of a train given its composition and the time it takes to cross a bridge. -/
theorem train_speed (num_carriages : ℕ) (carriage_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  num_carriages = 24 →
  carriage_length = 60 →
  bridge_length = 3.5 →
  crossing_time = 5 / 60 →
  let total_train_length := (num_carriages + 1 : ℝ) * carriage_length
  let total_distance := total_train_length / 1000 + bridge_length
  let speed := total_distance / crossing_time
  speed = 60 := by
    sorry


end NUMINAMATH_CALUDE_train_speed_l3148_314848


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l3148_314864

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l3148_314864


namespace NUMINAMATH_CALUDE_pencil_profit_calculation_pencil_profit_proof_l3148_314812

theorem pencil_profit_calculation (purchase_quantity : ℕ) (purchase_price : ℚ) 
  (selling_price : ℚ) (desired_profit : ℚ) (sold_quantity : ℕ) : Prop :=
  purchase_quantity = 2000 →
  purchase_price = 15/100 →
  selling_price = 30/100 →
  desired_profit = 150 →
  sold_quantity = 1500 →
  (sold_quantity : ℚ) * selling_price - (purchase_quantity : ℚ) * purchase_price = desired_profit

/-- Proof that selling 1500 pencils at $0.30 each results in a profit of $150 
    when 2000 pencils were purchased at $0.15 each. -/
theorem pencil_profit_proof :
  pencil_profit_calculation 2000 (15/100) (30/100) 150 1500 := by
  sorry

end NUMINAMATH_CALUDE_pencil_profit_calculation_pencil_profit_proof_l3148_314812


namespace NUMINAMATH_CALUDE_segment_length_proof_l3148_314899

theorem segment_length_proof (C D R S : ℝ) : 
  C < R ∧ R < S ∧ S < D →  -- R and S are on the same side of midpoint
  (R - C) / (D - R) = 3 / 5 →  -- R divides CD in ratio 3:5
  (S - C) / (D - S) = 2 / 3 →  -- S divides CD in ratio 2:3
  S - R = 5 →  -- Length of RS is 5
  D - C = 200 := by  -- Length of CD is 200
sorry


end NUMINAMATH_CALUDE_segment_length_proof_l3148_314899


namespace NUMINAMATH_CALUDE_range_of_S_l3148_314805

theorem range_of_S (x₁ x₂ x₃ x₄ : ℝ) 
  (h_nonneg : x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0) 
  (h_sum : x₁ + x₂ - x₃ + x₄ = 1) : 
  let S := 1 - (x₁^4 + x₂^4 + x₃^4 + x₄^4) - 
    6 * (x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄)
  0 ≤ S ∧ S ≤ 3/4 := by
sorry

end NUMINAMATH_CALUDE_range_of_S_l3148_314805


namespace NUMINAMATH_CALUDE_length_breadth_difference_l3148_314824

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  perimeter : ℝ
  fencing_rate : ℝ
  fencing_cost : ℝ

/-- Theorem stating the difference between length and breadth of the plot -/
theorem length_breadth_difference (plot : RectangularPlot)
  (h1 : plot.length = 61)
  (h2 : plot.perimeter * plot.fencing_rate = plot.fencing_cost)
  (h3 : plot.fencing_rate = 26.50)
  (h4 : plot.fencing_cost = 5300)
  (h5 : plot.perimeter = 2 * (plot.length + plot.breadth))
  (h6 : plot.length > plot.breadth) :
  plot.length - plot.breadth = 22 := by
  sorry

end NUMINAMATH_CALUDE_length_breadth_difference_l3148_314824


namespace NUMINAMATH_CALUDE_sin_plus_2cos_period_l3148_314840

open Real

/-- The function f(x) = sin x + 2cos x has a period of 2π. -/
theorem sin_plus_2cos_period : ∃ (k : ℝ), k > 0 ∧ ∀ x, sin x + 2 * cos x = sin (x + k) + 2 * cos (x + k) := by
  use 2 * π
  constructor
  · exact two_pi_pos
  · intro x
    sorry


end NUMINAMATH_CALUDE_sin_plus_2cos_period_l3148_314840


namespace NUMINAMATH_CALUDE_stratified_sampling_l3148_314850

-- Define the number of students in each grade
def freshmen : ℕ := 900
def sophomores : ℕ := 1200
def seniors : ℕ := 600

-- Define the total number of students
def total_students : ℕ := freshmen + sophomores + seniors

-- Define the sample size
def sample_size : ℕ := 135

-- Define the number of students to be sampled from each grade
def sampled_freshmen : ℕ := (sample_size * freshmen) / total_students
def sampled_sophomores : ℕ := (sample_size * sophomores) / total_students
def sampled_seniors : ℕ := (sample_size * seniors) / total_students

-- Theorem statement
theorem stratified_sampling :
  sampled_freshmen = 45 ∧
  sampled_sophomores = 60 ∧
  sampled_seniors = 30 ∧
  sampled_freshmen + sampled_sophomores + sampled_seniors = sample_size :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_l3148_314850


namespace NUMINAMATH_CALUDE_binomial_series_expansion_l3148_314818

theorem binomial_series_expansion (x : ℝ) (n : ℕ) (h : |x| < 1) :
  (1 / (1 - x))^n = 1 + ∑' k, (n + k - 1).choose (n - 1) * x^k :=
sorry

end NUMINAMATH_CALUDE_binomial_series_expansion_l3148_314818


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l3148_314822

theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 4 * p + 3 * q = 4.20)
  (h2 : 3 * p + 4 * q = 4.55) :
  p + q = 1.25 := by
sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l3148_314822


namespace NUMINAMATH_CALUDE_distinct_triangles_in_cube_l3148_314880

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles that can be formed by connecting three different vertices of a cube -/
def distinct_triangles : ℕ := Nat.choose cube_vertices triangle_vertices

theorem distinct_triangles_in_cube :
  distinct_triangles = 56 :=
sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_cube_l3148_314880


namespace NUMINAMATH_CALUDE_common_point_tangent_line_l3148_314839

theorem common_point_tangent_line (a : ℝ) (h_a : a > 0) :
  ∃ x : ℝ, x > 0 ∧ 
    a * Real.sqrt x = Real.log (Real.sqrt x) ∧
    (a / (2 * Real.sqrt x) = 1 / (2 * x)) →
  a = Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_common_point_tangent_line_l3148_314839
