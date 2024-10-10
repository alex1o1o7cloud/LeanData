import Mathlib

namespace inequality_proof_l2345_234596

theorem inequality_proof (a b c : ℝ) 
  (ha : a = (1/6) * Real.log 8)
  (hb : b = (1/2) * Real.log 5)
  (hc : c = Real.log (Real.sqrt 6) - Real.log (Real.sqrt 2)) :
  a < c ∧ c < b := by
  sorry

end inequality_proof_l2345_234596


namespace sector_angle_l2345_234555

/-- Given a circle with radius 12 meters and a sector with area 45.25714285714286 square meters,
    the central angle of the sector is 36 degrees. -/
theorem sector_angle (r : ℝ) (area : ℝ) (h1 : r = 12) (h2 : area = 45.25714285714286) :
  (area / (π * r^2)) * 360 = 36 := by
  sorry

end sector_angle_l2345_234555


namespace p_investment_calculation_l2345_234508

def investment_ratio (p_investment q_investment : ℚ) : ℚ := p_investment / q_investment

theorem p_investment_calculation (q_investment : ℚ) (profit_ratio : ℚ) :
  q_investment = 30000 →
  profit_ratio = 3 / 5 →
  ∃ p_investment : ℚ, 
    investment_ratio p_investment q_investment = profit_ratio ∧
    p_investment = 18000 := by
  sorry

end p_investment_calculation_l2345_234508


namespace unsatisfactory_fraction_is_8_25_l2345_234512

/-- Represents the grades in a class -/
structure GradeDistribution where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  f : Nat

/-- The grade distribution for the given class -/
def classGrades : GradeDistribution :=
  { a := 6, b := 5, c := 4, d := 2, f := 8 }

/-- The total number of students in the class -/
def totalStudents (grades : GradeDistribution) : Nat :=
  grades.a + grades.b + grades.c + grades.d + grades.f

/-- The number of students with unsatisfactory grades -/
def unsatisfactoryGrades (grades : GradeDistribution) : Nat :=
  grades.f

/-- Theorem: The fraction of unsatisfactory grades is 8/25 -/
theorem unsatisfactory_fraction_is_8_25 :
  (unsatisfactoryGrades classGrades : Rat) / (totalStudents classGrades) = 8 / 25 := by
  sorry

end unsatisfactory_fraction_is_8_25_l2345_234512


namespace vacation_cost_division_l2345_234588

theorem vacation_cost_division (total_cost : ℝ) (cost_difference : ℝ) : 
  total_cost = 480 →
  (total_cost / 4 = total_cost / 6 + cost_difference) →
  cost_difference = 40 →
  6 = (total_cost / (total_cost / 4 - cost_difference)) := by
sorry

end vacation_cost_division_l2345_234588


namespace geometric_sequence_term_number_l2345_234589

theorem geometric_sequence_term_number (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) = a n * (1/2))  -- geometric sequence with q = 1/2
  → a 1 = 1/2                         -- a₁ = 1/2
  → (∃ n : ℕ, a n = 1/32)             -- aₙ = 1/32 for some n
  → (∃ n : ℕ, a n = 1/32 ∧ n = 5) :=  -- prove that this n is 5
by sorry

end geometric_sequence_term_number_l2345_234589


namespace polygon_interior_exterior_angles_equality_l2345_234517

theorem polygon_interior_exterior_angles_equality (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 = 360) → 
  n = 4 := by
  sorry

end polygon_interior_exterior_angles_equality_l2345_234517


namespace restricted_choose_equals_44_l2345_234532

/-- The number of ways to choose r items from n items -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 cooks from 10 people with a restriction -/
def restrictedChoose : ℕ :=
  choose 10 2 - choose 2 2

theorem restricted_choose_equals_44 : restrictedChoose = 44 := by sorry

end restricted_choose_equals_44_l2345_234532


namespace complex_exponential_identity_l2345_234579

theorem complex_exponential_identity (n : ℕ) (hn : n > 0 ∧ n ≤ 500) (t : ℝ) :
  (Complex.exp (Complex.I * t))^n = Complex.exp (Complex.I * (n * t)) :=
by sorry

end complex_exponential_identity_l2345_234579


namespace rice_mixture_price_l2345_234531

/-- Given two types of rice with different weights and prices, 
    prove that the price of the second type can be determined 
    from the average price of the mixture. -/
theorem rice_mixture_price 
  (weight1 : ℝ) (price1 : ℝ) (weight2 : ℝ) (price2 : ℝ) (avg_price : ℝ)
  (h1 : weight1 = 8)
  (h2 : price1 = 16)
  (h3 : weight2 = 4)
  (h4 : avg_price = 18)
  (h5 : (weight1 * price1 + weight2 * price2) / (weight1 + weight2) = avg_price) :
  price2 = 22 := by
sorry

end rice_mixture_price_l2345_234531


namespace total_clothes_washed_l2345_234574

/-- Represents the number of clothes a person has -/
structure ClothesCount where
  whiteShirts : ℕ
  coloredShirts : ℕ
  shorts : ℕ
  pants : ℕ

/-- Calculates the total number of clothes for a person -/
def totalClothes (c : ClothesCount) : ℕ :=
  c.whiteShirts + c.coloredShirts + c.shorts + c.pants

/-- Cally's clothes count -/
def cally : ClothesCount :=
  { whiteShirts := 10
    coloredShirts := 5
    shorts := 7
    pants := 6 }

/-- Danny's clothes count -/
def danny : ClothesCount :=
  { whiteShirts := 6
    coloredShirts := 8
    shorts := 10
    pants := 6 }

/-- Theorem stating that the total number of clothes washed by Cally and Danny is 58 -/
theorem total_clothes_washed : totalClothes cally + totalClothes danny = 58 := by
  sorry

end total_clothes_washed_l2345_234574


namespace absolute_value_equation_solution_l2345_234542

theorem absolute_value_equation_solution : 
  ∃! x : ℝ, |x - 30| + |x - 25| = |2*x - 50| + 5 ∧ x = 32.5 := by sorry

end absolute_value_equation_solution_l2345_234542


namespace simplify_fraction_l2345_234584

theorem simplify_fraction : 15 * (18 / 11) * (-42 / 45) = -23 - (1 / 11) := by sorry

end simplify_fraction_l2345_234584


namespace polynomial_division_l2345_234520

theorem polynomial_division (x y : ℝ) (hx : x ≠ 0) :
  (15 * x^4 * y^2 - 12 * x^2 * y^3 - 3 * x^2) / (-3 * x^2) = -5 * x^2 * y^2 + 4 * y^3 + 1 := by
  sorry

end polynomial_division_l2345_234520


namespace simplify_expression_l2345_234500

theorem simplify_expression (a b : ℝ) :
  (17 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 42 * b) - 3 * (2 * a + 3 * b) = 14 * a + 30 * b := by
  sorry

end simplify_expression_l2345_234500


namespace spade_then_king_probability_l2345_234513

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of decks shuffled together -/
def num_decks : ℕ := 2

/-- The total number of cards after shuffling -/
def total_cards : ℕ := standard_deck_size * num_decks

/-- The number of spades in a standard deck -/
def spades_per_deck : ℕ := 13

/-- The number of kings in a standard deck -/
def kings_per_deck : ℕ := 4

/-- The probability of drawing a spade as the first card and a king as the second card -/
theorem spade_then_king_probability : 
  (spades_per_deck * num_decks) / total_cards * 
  (kings_per_deck * num_decks) / (total_cards - 1) = 103 / 5356 := by
  sorry

end spade_then_king_probability_l2345_234513


namespace square_area_from_diagonal_l2345_234534

theorem square_area_from_diagonal (d : ℝ) (h : d = 7) : 
  (d^2 / 2) = 24.5 := by
  sorry

end square_area_from_diagonal_l2345_234534


namespace prob_hit_both_l2345_234592

variable (p : ℝ)

-- Define the probability of hitting a single basket in 6 throws
def prob_hit_single (p : ℝ) : ℝ := 1 - (1 - p)^6

-- Define the probability of hitting at least one of two baskets in 6 throws
def prob_hit_at_least_one (p : ℝ) : ℝ := 1 - (1 - 2*p)^6

-- State the theorem
theorem prob_hit_both (hp : 0 ≤ p ∧ p ≤ 1/2) :
  prob_hit_single p + prob_hit_single p - prob_hit_at_least_one p = 1 - 2*(1 - p)^6 + (1 - 2*p)^6 := by
  sorry

end prob_hit_both_l2345_234592


namespace perfect_square_condition_l2345_234561

/-- A polynomial of the form x^2 + bx + c is a perfect square trinomial if and only if
    there exists a real number m such that b = 2m and c = m^2 -/
def is_perfect_square_trinomial (b c : ℝ) : Prop :=
  ∃ m : ℝ, b = 2 * m ∧ c = m^2

/-- The main theorem: x^2 + (a-1)x + 9 is a perfect square trinomial iff a = 7 or a = -5 -/
theorem perfect_square_condition (a : ℝ) :
  is_perfect_square_trinomial (a - 1) 9 ↔ a = 7 ∨ a = -5 := by
  sorry


end perfect_square_condition_l2345_234561


namespace smallest_sum_of_product_l2345_234551

theorem smallest_sum_of_product (a b c d e : ℕ+) : 
  a * b * c * d * e = Nat.factorial 12 → 
  (∀ w x y z v : ℕ+, w * x * y * z * v = Nat.factorial 12 → 
    a + b + c + d + e ≤ w + x + y + z + v) →
  a + b + c + d + e = 501 := by
  sorry

end smallest_sum_of_product_l2345_234551


namespace z_to_12_equals_one_l2345_234576

theorem z_to_12_equals_one :
  let z : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  z^12 = 1 := by sorry

end z_to_12_equals_one_l2345_234576


namespace no_valid_arrangement_l2345_234545

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i : Nat, i < arr.length - 1 →
    (10 * arr[i]! + arr[i+1]!) % 7 = 0

theorem no_valid_arrangement :
  ¬∃ arr : List Nat, arr.toFinset = {1, 2, 3, 4, 5, 6, 8, 9} ∧ is_valid_arrangement arr :=
sorry

end no_valid_arrangement_l2345_234545


namespace pyramid_height_l2345_234535

/-- The height of a triangular pyramid with a right-angled base and equal lateral edges -/
theorem pyramid_height (a b l : ℝ) (ha : 0 < a) (hb : 0 < b) (hl : 0 < l) :
  let h := (1 / 2 : ℝ) * Real.sqrt (4 * l^2 - a^2 - b^2)
  ∃ (h : ℝ), h > 0 ∧ h = (1 / 2 : ℝ) * Real.sqrt (4 * l^2 - a^2 - b^2) := by
  sorry

end pyramid_height_l2345_234535


namespace tangent_line_at_2_max_value_on_interval_l2345_234562

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 2

-- Statement for the tangent line
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ x y, y = f x → (x = 2 → y = m * (x - 2) + f 2) ∧
  (9 * x - y - 15 = 0 ↔ y = m * (x - 2) + f 2) :=
sorry

-- Statement for the maximum value
theorem max_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 3 :=
sorry

end tangent_line_at_2_max_value_on_interval_l2345_234562


namespace a5_equals_6_l2345_234557

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d

/-- The theorem stating that a5 = 6 in the given arithmetic sequence -/
theorem a5_equals_6 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : a 2 + a 8 = 12) :
  a 5 = 6 := by
  sorry

end a5_equals_6_l2345_234557


namespace water_bottles_used_second_game_l2345_234506

theorem water_bottles_used_second_game 
  (initial_cases : ℕ)
  (bottles_per_case : ℕ)
  (bottles_used_first_game : ℕ)
  (bottles_remaining_after_second_game : ℕ)
  (h1 : initial_cases = 10)
  (h2 : bottles_per_case = 20)
  (h3 : bottles_used_first_game = 70)
  (h4 : bottles_remaining_after_second_game = 20) :
  initial_cases * bottles_per_case - bottles_used_first_game - bottles_remaining_after_second_game = 110 :=
by sorry

end water_bottles_used_second_game_l2345_234506


namespace three_and_negative_three_are_opposite_l2345_234552

-- Definition of opposite numbers
def are_opposite (a b : ℝ) : Prop := (abs a = abs b) ∧ (a = -b)

-- Theorem to prove
theorem three_and_negative_three_are_opposite : are_opposite 3 (-3) := by
  sorry

end three_and_negative_three_are_opposite_l2345_234552


namespace andrew_final_stickers_l2345_234571

def total_stickers : ℕ := 1500
def ratio_sum : ℕ := 5

def initial_shares (i : Fin 3) : ℕ := 
  if i = 0 ∨ i = 1 then total_stickers / ratio_sum else 3 * (total_stickers / ratio_sum)

theorem andrew_final_stickers : 
  initial_shares 1 + (2/3 : ℚ) * initial_shares 2 = 900 := by sorry

end andrew_final_stickers_l2345_234571


namespace remainder_3m_mod_5_l2345_234591

theorem remainder_3m_mod_5 (m : ℤ) (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end remainder_3m_mod_5_l2345_234591


namespace max_cube_volume_in_tetrahedron_l2345_234515

/-- Regular tetrahedron with edge length 2 -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_eq : edge_length = 2

/-- Cube placed inside the tetrahedron -/
structure InsideCube where
  side_length : ℝ
  bottom_face_parallel : Prop
  top_vertices_touch : Prop

/-- The maximum volume of the cube inside the tetrahedron -/
def max_cube_volume (t : RegularTetrahedron) (c : InsideCube) : ℝ :=
  c.side_length ^ 3

/-- Theorem stating the maximum volume of the cube -/
theorem max_cube_volume_in_tetrahedron (t : RegularTetrahedron) (c : InsideCube) :
  max_cube_volume t c = 8 * Real.sqrt 3 / 243 :=
sorry

end max_cube_volume_in_tetrahedron_l2345_234515


namespace parabola_f_value_l2345_234524

-- Define the parabola equation
def parabola (d e f y : ℝ) : ℝ := d * y^2 + e * y + f

-- Theorem statement
theorem parabola_f_value :
  ∀ d e f : ℝ,
  -- Vertex condition
  (∀ y : ℝ, parabola d e f (-3) = 2) ∧
  -- Point (7, 0) condition
  parabola d e f 0 = 7 →
  -- Conclusion: f = 7
  f = 7 := by
  sorry

end parabola_f_value_l2345_234524


namespace tangent_parabola_circle_l2345_234543

/-- Theorem: Tangent Line to Parabola Touching Circle -/
theorem tangent_parabola_circle (r : ℝ) (hr : r > 0) :
  ∃ (x y : ℝ),
    -- Point P(x, y) lies on the parabola
    y = (1/4) * x^2 ∧
    -- Point P(x, y) lies on the circle
    (x - 1)^2 + (y - 2)^2 = r^2 ∧
    -- The tangent line to the parabola at P touches the circle
    ∃ (m : ℝ),
      -- m is the slope of the tangent line to the parabola at P
      m = (1/2) * x ∧
      -- The tangent line touches the circle (perpendicular to radius)
      m * ((y - 2) / (x - 1)) = -1
  → r = Real.sqrt 2 :=
by sorry

end tangent_parabola_circle_l2345_234543


namespace inverse_function_range_l2345_234536

def is_inverse_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) * f (a - x) = 1

theorem inverse_function_range 
  (f : ℝ → ℝ) 
  (h0 : is_inverse_function f 0)
  (h1 : is_inverse_function f 1)
  (h_range : ∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 1 2) :
  ∀ x ∈ Set.Icc (-2016) 2016, f x ∈ Set.Icc (1/2) 2 :=
sorry

end inverse_function_range_l2345_234536


namespace jump_ratio_l2345_234527

def hattie_first_round : ℕ := 180
def lorelei_first_round : ℕ := (3 * hattie_first_round) / 4
def total_jumps : ℕ := 605

def hattie_second_round : ℕ := (total_jumps - hattie_first_round - lorelei_first_round - 50) / 2

theorem jump_ratio : 
  (hattie_second_round : ℚ) / hattie_first_round = 2 / 3 := by sorry

end jump_ratio_l2345_234527


namespace quadratic_equation_roots_l2345_234587

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (r₁ + r₂ = 12 ∧ |r₁ - r₂| = 10) ↔ (a = 1 ∧ b = -12 ∧ c = 11) :=
by sorry

end quadratic_equation_roots_l2345_234587


namespace games_attended_l2345_234558

theorem games_attended (total : ℕ) (missed : ℕ) (attended : ℕ) : 
  total = 12 → missed = 7 → attended = total - missed → attended = 5 := by
  sorry

end games_attended_l2345_234558


namespace outfits_count_l2345_234594

/-- The number of shirts available. -/
def num_shirts : ℕ := 5

/-- The number of pairs of pants available. -/
def num_pants : ℕ := 3

/-- The number of ties available. -/
def num_ties : ℕ := 2

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_pants * num_ties

/-- Theorem stating that the total number of possible outfits is 30. -/
theorem outfits_count : total_outfits = 30 := by
  sorry

end outfits_count_l2345_234594


namespace abs_g_one_equals_31_l2345_234547

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- Condition that the absolute value of g at specific points is 24 -/
def SatisfiesCondition (g : ThirdDegreePolynomial) : Prop :=
  |g (-1)| = 24 ∧ |g 0| = 24 ∧ |g 2| = 24 ∧ |g 4| = 24 ∧ |g 5| = 24 ∧ |g 8| = 24

/-- The main theorem -/
theorem abs_g_one_equals_31 (g : ThirdDegreePolynomial) 
  (h : SatisfiesCondition g) : |g 1| = 31 := by
  sorry

end abs_g_one_equals_31_l2345_234547


namespace total_students_in_line_l2345_234504

/-- The number of students in a line, given specific positions of Hoseok and Yoongi -/
def number_of_students (left_of_hoseok : ℕ) (between_hoseok_yoongi : ℕ) (right_of_yoongi : ℕ) : ℕ :=
  left_of_hoseok + 1 + between_hoseok_yoongi + 1 + right_of_yoongi

/-- Theorem stating that the total number of students in the line is 22 -/
theorem total_students_in_line : 
  number_of_students 9 5 6 = 22 := by
  sorry

end total_students_in_line_l2345_234504


namespace abby_damon_weight_l2345_234578

theorem abby_damon_weight 
  (a b c d : ℝ)  -- Weights of Abby, Bart, Cindy, and Damon
  (h1 : a + b = 280)  -- Abby and Bart's combined weight
  (h2 : b + c = 255)  -- Bart and Cindy's combined weight
  (h3 : c + d = 290)  -- Cindy and Damon's combined weight
  : a + d = 315 := by
  sorry

end abby_damon_weight_l2345_234578


namespace initial_balloons_eq_sum_l2345_234541

/-- The number of green balloons Fred initially had -/
def initial_balloons : ℕ := sorry

/-- The number of green balloons Fred gave to Sandy -/
def balloons_given : ℕ := 221

/-- The number of green balloons Fred has left -/
def balloons_left : ℕ := 488

/-- Theorem stating that the initial number of balloons is equal to the sum of balloons given away and balloons left -/
theorem initial_balloons_eq_sum : initial_balloons = balloons_given + balloons_left := by sorry

end initial_balloons_eq_sum_l2345_234541


namespace pens_given_to_sharon_l2345_234597

/-- The number of pens given to Sharon -/
def pens_to_sharon (initial : ℕ) (from_mike : ℕ) (final : ℕ) : ℕ :=
  2 * (initial + from_mike) - final

theorem pens_given_to_sharon :
  pens_to_sharon 5 20 40 = 10 := by
  sorry

end pens_given_to_sharon_l2345_234597


namespace coefficient_x_squared_l2345_234553

theorem coefficient_x_squared (p q : Polynomial ℤ) : 
  p = X^3 - 4*X^2 + 6*X - 2 →
  q = 3*X^2 - 2*X + 5 →
  (p * q).coeff 2 = -38 := by
sorry

end coefficient_x_squared_l2345_234553


namespace bus_performance_analysis_l2345_234590

structure BusCompany where
  name : String
  onTime : ℕ
  notOnTime : ℕ

def totalBuses (company : BusCompany) : ℕ := company.onTime + company.notOnTime

def onTimeProbability (company : BusCompany) : ℚ :=
  company.onTime / totalBuses company

def kSquared (companyA companyB : BusCompany) : ℚ :=
  let n := totalBuses companyA + totalBuses companyB
  let a := companyA.onTime
  let b := companyA.notOnTime
  let c := companyB.onTime
  let d := companyB.notOnTime
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def companyA : BusCompany := ⟨"A", 240, 20⟩
def companyB : BusCompany := ⟨"B", 210, 30⟩

theorem bus_performance_analysis :
  (onTimeProbability companyA = 12/13) ∧ 
  (onTimeProbability companyB = 7/8) ∧ 
  (kSquared companyA companyB > 2706/1000) := by
  sorry

end bus_performance_analysis_l2345_234590


namespace rational_expression_iff_perfect_square_l2345_234521

theorem rational_expression_iff_perfect_square (x : ℝ) :
  ∃ (q : ℚ), x + Real.sqrt (x^2 + 9) - 1 / (x + Real.sqrt (x^2 + 9)) = q ↔ 
  ∃ (n : ℕ), x^2 + 9 = n^2 := by
sorry

end rational_expression_iff_perfect_square_l2345_234521


namespace egg_distribution_l2345_234563

theorem egg_distribution (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) : 
  total_eggs = 8 → num_groups = 4 → eggs_per_group = total_eggs / num_groups → eggs_per_group = 2 := by
  sorry

end egg_distribution_l2345_234563


namespace fraction_of_powers_equals_81_l2345_234518

theorem fraction_of_powers_equals_81 : (75000 ^ 4) / (25000 ^ 4) = 81 := by
  sorry

end fraction_of_powers_equals_81_l2345_234518


namespace percentage_problem_l2345_234582

theorem percentage_problem (x : ℝ) :
  (0.15 * 0.30 * 0.50 * x = 117) → (x = 5200) :=
by sorry

end percentage_problem_l2345_234582


namespace no_positive_integer_solution_l2345_234568

theorem no_positive_integer_solution :
  ¬ ∃ (n : ℕ+) (p : ℕ), Nat.Prime p ∧ n.val^2 - 45*n.val + 520 = p := by
  sorry

end no_positive_integer_solution_l2345_234568


namespace sum_first_60_eq_1830_l2345_234516

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the first 60 natural numbers is 1830 -/
theorem sum_first_60_eq_1830 : sum_first_n 60 = 1830 := by
  sorry

end sum_first_60_eq_1830_l2345_234516


namespace largest_perfect_square_factor_7560_l2345_234511

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

theorem largest_perfect_square_factor_7560 :
  largest_perfect_square_factor 7560 = 36 := by
  sorry

end largest_perfect_square_factor_7560_l2345_234511


namespace relationship_between_variables_l2345_234554

-- Define variables
variable (a b c d : ℝ)
variable (x y q z : ℝ)

-- Define the theorem
theorem relationship_between_variables 
  (h1 : a^(3*x) = c^(2*q)) 
  (h2 : c^(2*q) = b)
  (h3 : c^(4*y) = a^(5*z))
  (h4 : a^(5*z) = d) :
  5*q*z = 6*x*y := by
  sorry

end relationship_between_variables_l2345_234554


namespace tan_120_degrees_l2345_234580

theorem tan_120_degrees : Real.tan (120 * π / 180) = -Real.sqrt 3 := by
  sorry

end tan_120_degrees_l2345_234580


namespace geometric_sequence_common_ratio_l2345_234502

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a2 : a 2 = 2) 
  (h_a5 : a 5 = 1/4) : 
  a 1 / a 0 = 1/2 := by
sorry

end geometric_sequence_common_ratio_l2345_234502


namespace point_on_graph_l2345_234519

/-- The function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- The point we're checking -/
def point : ℝ × ℝ := (1, -1)

/-- Theorem: The point (1, -1) lies on the graph of f(x) = -2x + 1 -/
theorem point_on_graph : f point.1 = point.2 := by
  sorry

end point_on_graph_l2345_234519


namespace det_of_matrix_l2345_234593

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; -3, 6]

theorem det_of_matrix : Matrix.det matrix = 36 := by sorry

end det_of_matrix_l2345_234593


namespace basic_computer_price_l2345_234501

theorem basic_computer_price
  (total_price : ℝ)
  (price_difference : ℝ)
  (printer_ratio : ℝ)
  (h1 : total_price = 2500)
  (h2 : price_difference = 500)
  (h3 : printer_ratio = 1/4)
  : ∃ (basic_computer_price printer_price : ℝ),
    basic_computer_price + printer_price = total_price ∧
    printer_price = printer_ratio * (basic_computer_price + price_difference + printer_price) ∧
    basic_computer_price = 1750 :=
by sorry

end basic_computer_price_l2345_234501


namespace exist_triangle_area_le_two_l2345_234503

-- Define a lattice point
def LatticePoint := ℤ × ℤ

-- Define the condition for points within the square region
def WithinSquare (p : LatticePoint) : Prop :=
  |p.1| ≤ 2 ∧ |p.2| ≤ 2

-- Define a function to calculate the area of a triangle given three points
def TriangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Define the property of three points not being collinear
def NotCollinear (p1 p2 p3 : LatticePoint) : Prop :=
  TriangleArea p1 p2 p3 ≠ 0

-- Main theorem
theorem exist_triangle_area_le_two 
  (points : Fin 6 → LatticePoint)
  (h1 : ∀ i, WithinSquare (points i))
  (h2 : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → NotCollinear (points i) (points j) (points k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ TriangleArea (points i) (points j) (points k) ≤ 2 :=
sorry

end exist_triangle_area_le_two_l2345_234503


namespace norm_equals_5_sqrt_5_l2345_234575

def vector : Fin 2 → ℝ
  | 0 => 3
  | 1 => 1

theorem norm_equals_5_sqrt_5 (k : ℝ) : 
  ∃ (v : Fin 2 → ℝ), v 0 = -5 ∧ v 1 = 6 ∧
  (‖(k • vector - v)‖ = 5 * Real.sqrt 5) ↔ 
  (k = (-9 + Real.sqrt 721) / 10 ∨ k = (-9 - Real.sqrt 721) / 10) :=
by sorry

end norm_equals_5_sqrt_5_l2345_234575


namespace sum_first_six_primes_mod_seventh_prime_l2345_234540

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_first_six_primes_mod_seventh_prime :
  (first_six_primes.sum % seventh_prime) = 7 := by
  sorry

end sum_first_six_primes_mod_seventh_prime_l2345_234540


namespace scrap_rate_cost_relationship_l2345_234505

/-- Represents the regression line equation for pig iron cost -/
def regression_line (x : ℝ) : ℝ := 256 + 3 * x

/-- Theorem stating the relationship between scrap rate increase and cost increase -/
theorem scrap_rate_cost_relationship (x : ℝ) :
  regression_line (x + 1) - regression_line x = 3 := by
  sorry

end scrap_rate_cost_relationship_l2345_234505


namespace revenue_decrease_l2345_234528

theorem revenue_decrease (R : ℝ) (h1 : R > 0) : 
  let projected_revenue := 1.4 * R
  let actual_revenue := 0.5 * projected_revenue
  let percent_decrease := (R - actual_revenue) / R * 100
  percent_decrease = 30 := by
  sorry

end revenue_decrease_l2345_234528


namespace toy_store_problem_l2345_234581

/-- Toy Store Problem -/
theorem toy_store_problem 
  (first_batch_cost second_batch_cost : ℝ)
  (quantity_ratio : ℝ)
  (cost_increase : ℝ)
  (min_profit : ℝ)
  (h1 : first_batch_cost = 2500)
  (h2 : second_batch_cost = 4500)
  (h3 : quantity_ratio = 1.5)
  (h4 : cost_increase = 10)
  (h5 : min_profit = 1750) :
  ∃ (first_batch_cost_per_set min_selling_price : ℝ),
    first_batch_cost_per_set = 50 ∧
    min_selling_price = 70 ∧
    (quantity_ratio * first_batch_cost / first_batch_cost_per_set) * min_selling_price +
    (first_batch_cost / first_batch_cost_per_set) * min_selling_price -
    first_batch_cost - second_batch_cost ≥ min_profit :=
by sorry

end toy_store_problem_l2345_234581


namespace marker_cost_l2345_234549

theorem marker_cost (total_students : ℕ) (buyers : ℕ) (markers_per_student : ℕ) (marker_cost : ℕ) :
  total_students = 24 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  markers_per_student > 1 →
  marker_cost > markers_per_student →
  buyers * marker_cost * markers_per_student = 924 →
  marker_cost = 11 :=
by sorry

end marker_cost_l2345_234549


namespace min_value_of_cubic_function_l2345_234573

/-- Given a function f(x) = 2x^3 - 6x^2 + a, where a is a constant,
    prove that if the maximum value of f(x) on the interval [-2, 2] is 3,
    then the minimum value of f(x) on [-2, 2] is -37. -/
theorem min_value_of_cubic_function (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^3 - 6 * x^2 + a
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 3) ∧ (∃ x ∈ Set.Icc (-2) 2, f x = 3) →
  (∃ x ∈ Set.Icc (-2) 2, f x = -37) ∧ (∀ x ∈ Set.Icc (-2) 2, f x ≥ -37) :=
by sorry

end min_value_of_cubic_function_l2345_234573


namespace factor_expression_l2345_234548

theorem factor_expression (x y : ℝ) : 3 * x^2 - 75 * y^2 = 3 * (x + 5*y) * (x - 5*y) := by
  sorry

end factor_expression_l2345_234548


namespace inequality_proof_l2345_234544

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l2345_234544


namespace paige_recycled_amount_l2345_234570

/-- The number of pounds recycled per point earned -/
def pounds_per_point : ℕ := 4

/-- The number of pounds recycled by Paige's friends -/
def friends_recycled : ℕ := 2

/-- The total number of points earned -/
def total_points : ℕ := 4

/-- The number of pounds Paige recycled -/
def paige_recycled : ℕ := 14

theorem paige_recycled_amount :
  paige_recycled = total_points * pounds_per_point - friends_recycled := by
  sorry

end paige_recycled_amount_l2345_234570


namespace binomial_product_and_evaluation_l2345_234526

theorem binomial_product_and_evaluation :
  ∀ x : ℝ,
  (4 * x + 3) * (2 * x - 6) = 8 * x^2 - 18 * x - 18 ∧
  (8 * (-1)^2 - 18 * (-1) - 18) = 8 := by
  sorry

end binomial_product_and_evaluation_l2345_234526


namespace no_solution_exists_l2345_234565

theorem no_solution_exists : ¬ ∃ (a b : ℝ), a^2 + 3*b^2 + 2 = 3*a*b := by sorry

end no_solution_exists_l2345_234565


namespace range_of_m_l2345_234585

theorem range_of_m (m : ℝ) : 
  (∃ x₀ ∈ Set.Icc 1 2, x₀^2 - m*x₀ + 4 > 0) ↔ m < 5 :=
sorry

end range_of_m_l2345_234585


namespace investment_years_equals_three_l2345_234539

/-- Calculates the number of years for which a principal is invested, given the interest rate,
    principal amount, and the difference between the principal and interest. -/
def calculate_investment_years (rate : ℚ) (principal : ℚ) (principal_minus_interest : ℚ) : ℚ :=
  (principal - principal_minus_interest) / (principal * rate / 100)

theorem investment_years_equals_three :
  let rate : ℚ := 12
  let principal : ℚ := 9200
  let principal_minus_interest : ℚ := 5888
  calculate_investment_years rate principal principal_minus_interest = 3 := by
  sorry

end investment_years_equals_three_l2345_234539


namespace polynomial_roots_product_l2345_234569

theorem polynomial_roots_product (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℤ, x^3 + a*x^2 + b*x + 6*a = (x - r)^2 * (x - s)) ∧ 
   r ≠ s) → 
  |a * b| = 546 := by
  sorry

end polynomial_roots_product_l2345_234569


namespace solution_value_l2345_234538

theorem solution_value (a : ℝ) : (∃ x : ℝ, x = 1 ∧ a * x + 2 * x = 3) → a = 1 := by
  sorry

end solution_value_l2345_234538


namespace odd_prime_sqrt_integer_l2345_234509

theorem odd_prime_sqrt_integer (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) 
  (h_pos : k > 0) (h_sqrt : ∃ n : ℕ, n > 0 ∧ n^2 = k^2 - p*k) : 
  k = (p + 1)^2 / 4 := by
sorry

end odd_prime_sqrt_integer_l2345_234509


namespace chord_length_is_sqrt_34_l2345_234546

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + m = 0
def l (x y : ℝ) : Prop := x + y = 0

-- Define external tangency
def externally_tangent (m : ℝ) : Prop :=
  ∃ x y, C₁ x y ∧ C₂ x y m ∧ (x - 0)^2 + (y - 0)^2 = (2 + Real.sqrt (25 - m))^2

-- Theorem statement
theorem chord_length_is_sqrt_34 (m : ℝ) :
  externally_tangent m →
  ∃ x₁ y₁ x₂ y₂,
    C₂ x₁ y₁ m ∧ C₂ x₂ y₂ m ∧
    l x₁ y₁ ∧ l x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 34 :=
by sorry

end chord_length_is_sqrt_34_l2345_234546


namespace largest_power_dividing_factorial_squared_l2345_234577

theorem largest_power_dividing_factorial_squared (p : ℕ) (hp : Prime p) :
  (∃ k : ℕ, (p ^ k : ℕ) ∣ (p^2).factorial ∧ 
   ∀ m : ℕ, (p ^ m : ℕ) ∣ (p^2).factorial → m ≤ k) ↔ 
  (∃ k : ℕ, k = p + 1 ∧ (p ^ k : ℕ) ∣ (p^2).factorial ∧ 
   ∀ m : ℕ, (p ^ m : ℕ) ∣ (p^2).factorial → m ≤ k) :=
by sorry

end largest_power_dividing_factorial_squared_l2345_234577


namespace lcm_gcd_product_l2345_234525

theorem lcm_gcd_product (a b : ℕ) (ha : a = 28) (hb : b = 45) :
  (Nat.lcm a b) * (Nat.gcd a b) = a * b := by
  sorry

end lcm_gcd_product_l2345_234525


namespace arithmetic_sequence_common_difference_l2345_234559

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 4 + a 8 = 10) 
  (h_term : a 10 = 6) : 
  ∃ d : ℚ, d = 1/4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l2345_234559


namespace workers_count_l2345_234533

theorem workers_count (total_work : ℕ) : ∃ (workers : ℕ),
  (workers * 65 = total_work) ∧
  ((workers + 10) * 55 = total_work) ∧
  (workers = 55) := by
sorry

end workers_count_l2345_234533


namespace cats_in_sacks_l2345_234598

theorem cats_in_sacks (cat_prices sack_prices : Finset ℕ) : 
  cat_prices.card = 20 →
  sack_prices.card = 20 →
  (∀ p ∈ cat_prices, 1200 ≤ p ∧ p ≤ 1500) →
  (∀ p ∈ sack_prices, 10 ≤ p ∧ p ≤ 100) →
  cat_prices.toList.Nodup →
  sack_prices.toList.Nodup →
  ∃ (c1 c2 : ℕ) (s1 s2 : ℕ),
    c1 ∈ cat_prices ∧ 
    c2 ∈ cat_prices ∧ 
    s1 ∈ sack_prices ∧ 
    s2 ∈ sack_prices ∧
    c1 ≠ c2 ∧ 
    s1 ≠ s2 ∧ 
    c1 + s1 = c2 + s2 :=
by sorry

end cats_in_sacks_l2345_234598


namespace max_power_under_500_l2345_234510

theorem max_power_under_500 (a b : ℕ) (ha : a > 0) (hb : b > 2) (h_less_500 : a^b < 500) :
  ∃ (a_max b_max : ℕ),
    a_max > 0 ∧ b_max > 2 ∧ a_max^b_max < 500 ∧
    ∀ (x y : ℕ), x > 0 → y > 2 → x^y < 500 → x^y ≤ a_max^b_max ∧
    a_max + b_max = 8 := by
  sorry

end max_power_under_500_l2345_234510


namespace xyz_value_l2345_234529

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 80 := by
sorry

end xyz_value_l2345_234529


namespace necessary_but_not_sufficient_condition_for_greater_than_two_l2345_234537

theorem necessary_but_not_sufficient_condition_for_greater_than_two (a : ℝ) :
  (a ≥ 2 → a > 2 → True) ∧ ¬(a ≥ 2 → a > 2) :=
by sorry

end necessary_but_not_sufficient_condition_for_greater_than_two_l2345_234537


namespace cone_sphere_ratio_l2345_234564

theorem cone_sphere_ratio (r h : ℝ) (h_pos : 0 < r) : 
  (1 / 3 : ℝ) * (4 / 3 * π * r^3) = (1 / 3 : ℝ) * π * r^2 * h → h / r = 4 / 3 := by
  sorry

end cone_sphere_ratio_l2345_234564


namespace more_girls_than_boys_l2345_234507

theorem more_girls_than_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 466 →
  boys = 127 →
  girls > boys →
  total = girls + boys →
  girls - boys = 212 :=
by sorry

end more_girls_than_boys_l2345_234507


namespace binomial_identity_l2345_234583

theorem binomial_identity (n k : ℕ) (hn : n > 1) (hk : k > 1) (hkn : k ≤ n) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end binomial_identity_l2345_234583


namespace f_properties_l2345_234550

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def f_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≠ Real.pi * ↑(Int.floor (x / Real.pi)) →
         y ≠ Real.pi * ↑(Int.floor (y / Real.pi)) →
         f (x - y) = (f x * f y + 1) / (f y - f x)

theorem f_properties (f : ℝ → ℝ) 
  (h_eq : f_equation f)
  (h_f1 : f 1 = 1)
  (h_pos : ∀ x, 0 < x → x < 2 → f x > 0) :
  is_odd f ∧ 
  f 2 = 0 ∧ 
  f 3 = -1 ∧
  (∀ x, 2 ≤ x → x ≤ 3 → f x ≤ 0) ∧
  (∀ x, 2 ≤ x → x ≤ 3 → f x ≥ -1) :=
sorry

end f_properties_l2345_234550


namespace solution_set_l2345_234556

def satisfies_equations (x y : ℝ) : Prop :=
  y^2 - y*x^2 = 0 ∧ x^5 + x^4 = 0

theorem solution_set :
  ∀ x y : ℝ, satisfies_equations x y ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = 1) :=
by sorry

end solution_set_l2345_234556


namespace smallest_even_triangle_perimeter_l2345_234586

/-- Represents a triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  n : ℕ
  side1 : ℕ := 2 * n
  side2 : ℕ := 2 * n + 2
  side3 : ℕ := 2 * n + 4

/-- Checks if the given EvenTriangle satisfies the triangle inequality -/
def is_valid (t : EvenTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem: The smallest possible perimeter of a valid EvenTriangle is 18 -/
theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), is_valid t ∧ perimeter t = 18 ∧
  ∀ (t' : EvenTriangle), is_valid t' → perimeter t' ≥ 18 :=
sorry

end smallest_even_triangle_perimeter_l2345_234586


namespace partnership_investment_l2345_234560

theorem partnership_investment (b c total_profit a_profit : ℕ) 
  (hb : b = 4200)
  (hc : c = 10500)
  (htotal : total_profit = 14200)
  (ha_profit : a_profit = 4260) :
  ∃ a : ℕ, a = 6600 ∧ a_profit / total_profit = a / (a + b + c) :=
sorry

end partnership_investment_l2345_234560


namespace certain_number_proof_l2345_234522

theorem certain_number_proof : ∃ x : ℕ, x * 12 = 173 * 240 ∧ x = 3460 := by
  sorry

end certain_number_proof_l2345_234522


namespace tie_cost_l2345_234567

theorem tie_cost (pants_cost shirt_cost paid change : ℕ) 
  (h1 : pants_cost = 140)
  (h2 : shirt_cost = 43)
  (h3 : paid = 200)
  (h4 : change = 2) :
  paid - change - (pants_cost + shirt_cost) = 15 := by
sorry

end tie_cost_l2345_234567


namespace light_blocks_count_is_twenty_l2345_234572

/-- Represents a tower with light colored blocks -/
structure LightTower where
  central_column_height : ℕ
  outer_columns_count : ℕ
  outer_column_height : ℕ

/-- Calculates the total number of light colored blocks in the tower -/
def total_light_blocks (tower : LightTower) : ℕ :=
  tower.central_column_height + tower.outer_columns_count * tower.outer_column_height

/-- Theorem stating that the total number of light colored blocks in the specific tower is 20 -/
theorem light_blocks_count_is_twenty :
  ∃ (tower : LightTower),
    tower.central_column_height = 4 ∧
    tower.outer_columns_count = 8 ∧
    tower.outer_column_height = 2 ∧
    total_light_blocks tower = 20 := by
  sorry


end light_blocks_count_is_twenty_l2345_234572


namespace rationalize_denominator_l2345_234595

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℤ), 
    F > 0 ∧
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) = 
      (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    A = 13 ∧ B = 9 ∧ C = -3 ∧ D = -2 ∧ E = 165 ∧ F = 51 :=
by sorry

end rationalize_denominator_l2345_234595


namespace temperature_theorem_l2345_234530

def temperature_problem (temp_ny temp_miami temp_sd temp_phoenix : ℝ) : Prop :=
  temp_ny = 80 ∧
  temp_miami = temp_ny + 10 ∧
  temp_sd = temp_miami + 25 ∧
  temp_phoenix = temp_sd * 1.15 ∧
  (temp_ny + temp_miami + temp_sd + temp_phoenix) / 4 = 104.3125

theorem temperature_theorem :
  ∃ temp_ny temp_miami temp_sd temp_phoenix : ℝ,
    temperature_problem temp_ny temp_miami temp_sd temp_phoenix := by
  sorry

end temperature_theorem_l2345_234530


namespace expression_evaluates_to_one_l2345_234599

theorem expression_evaluates_to_one :
  (100^2 - 7^2) / (70^2 - 11^2) * ((70 - 11) * (70 + 11)) / ((100 - 7) * (100 + 7)) = 1 := by
  sorry

end expression_evaluates_to_one_l2345_234599


namespace reflect_P_x_axis_l2345_234523

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (3, -2)

/-- Theorem: Reflecting P(3,-2) across the x-axis results in (3,2) -/
theorem reflect_P_x_axis : reflect_x P = (3, 2) := by
  sorry

end reflect_P_x_axis_l2345_234523


namespace population_scientific_notation_l2345_234514

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem population_scientific_notation :
  toScientificNotation (141260 * 1000000) =
    ScientificNotation.mk 1.4126 5 (by norm_num) :=
  sorry

end population_scientific_notation_l2345_234514


namespace parallel_lines_circle_chord_l2345_234566

/-- Given three equally spaced parallel lines intersecting a circle, creating chords of lengths 38, 38, and 34, the distance between two adjacent parallel lines is 6. -/
theorem parallel_lines_circle_chord (r : ℝ) : 
  let chord1 : ℝ := 38
  let chord2 : ℝ := 38
  let chord3 : ℝ := 34
  let d : ℝ := 6
  38 * r^2 = 722 + (19/4) * d^2 ∧ 
  34 * r^2 = 578 + (153/4) * d^2 →
  d = 6 := by
sorry

end parallel_lines_circle_chord_l2345_234566
