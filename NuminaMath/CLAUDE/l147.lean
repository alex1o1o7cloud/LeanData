import Mathlib

namespace parallel_vector_sum_diff_l147_14793

/-- Given two 2D vectors a and b, if a + b is parallel to a - b, then the first component of a is -4/3. -/
theorem parallel_vector_sum_diff (a b : ℝ × ℝ) :
  a.1 = m ∧ a.2 = 2 ∧ b = (2, -3) →
  (∃ k : ℝ, k ≠ 0 ∧ (a + b) = k • (a - b)) →
  m = -4/3 := by sorry

end parallel_vector_sum_diff_l147_14793


namespace binary_110011_equals_51_l147_14706

-- Define the binary number as a list of bits (0 or 1)
def binary_number : List Nat := [1, 1, 0, 0, 1, 1]

-- Define the function to convert binary to decimal
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

-- Theorem to prove
theorem binary_110011_equals_51 :
  binary_to_decimal binary_number = 51 := by
  sorry

end binary_110011_equals_51_l147_14706


namespace triangle_problem_l147_14713

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B →
  a + c = 6 →
  (1/2) * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 →
  B = π / 3 ∧ b = 3 * Real.sqrt 2 := by
sorry

end triangle_problem_l147_14713


namespace prob_two_dice_shows_two_l147_14721

def num_sides : ℕ := 8

def prob_at_least_one_two (n : ℕ) : ℚ :=
  1 - ((n - 1) / n)^2

theorem prob_two_dice_shows_two :
  prob_at_least_one_two num_sides = 15 / 64 := by
  sorry

end prob_two_dice_shows_two_l147_14721


namespace quadratic_point_relationship_l147_14792

/-- The quadratic function f(x) = x² + 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

theorem quadratic_point_relationship (c : ℝ) :
  let y₁ := f c (-4)
  let y₂ := f c (-3)
  let y₃ := f c 1
  y₂ < y₁ ∧ y₁ < y₃ := by sorry

end quadratic_point_relationship_l147_14792


namespace ratio_of_system_l147_14762

theorem ratio_of_system (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end ratio_of_system_l147_14762


namespace distance_between_intersections_l147_14700

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := ∃ θ : ℝ, x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sin θ

-- Define the ray
def ray (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x ∧ x ≥ 0

-- Define the intersection points
def intersectionC₁ (x y : ℝ) : Prop := C₁ x y ∧ ray x y
def intersectionC₂ (x y : ℝ) : Prop := C₂ x y ∧ ray x y

-- Theorem statement
theorem distance_between_intersections :
  ∃ (A B : ℝ × ℝ),
    intersectionC₁ A.1 A.2 ∧
    intersectionC₂ B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 - 2 * Real.sqrt 10 / 5 :=
sorry

end distance_between_intersections_l147_14700


namespace lawrence_county_kids_at_home_l147_14771

/-- The number of kids from Lawrence county who stay home during summer break -/
def kids_stay_home (total_kids : ℕ) (kids_at_camp : ℕ) : ℕ :=
  total_kids - kids_at_camp

/-- Proof that 590796 kids from Lawrence county stay home during summer break -/
theorem lawrence_county_kids_at_home : 
  kids_stay_home 1201565 610769 = 590796 := by
  sorry

end lawrence_county_kids_at_home_l147_14771


namespace book_area_l147_14733

/-- The area of a rectangular book with length 5 inches and width 10 inches is 50 square inches. -/
theorem book_area : 
  let length : ℝ := 5
  let width : ℝ := 10
  let area := length * width
  area = 50 := by sorry

end book_area_l147_14733


namespace g_tan_squared_l147_14722

open Real

noncomputable def g (x : ℝ) : ℝ := 1 / ((x - 1) / x)

theorem g_tan_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/2) :
  g (tan t ^ 2) = tan t ^ 2 - tan t ^ 4 :=
by sorry

end g_tan_squared_l147_14722


namespace quadratic_equation_m_l147_14794

/-- Given that (m+3)x^(m^2-7) + mx - 2 = 0 is a quadratic equation in x, prove that m = 3 -/
theorem quadratic_equation_m (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 3) * x^(m^2 - 7) + m * x - 2 = a * x^2 + b * x + c) →
  m = 3 :=
by sorry

end quadratic_equation_m_l147_14794


namespace smallest_positive_root_of_f_l147_14753

open Real

theorem smallest_positive_root_of_f (f : ℝ → ℝ) :
  (∀ x, f x = sin x + 2 * cos x + 3 * tan x) →
  (∃ x ∈ Set.Ioo 3 4, f x = 0) ∧
  (∀ x ∈ Set.Ioo 0 3, f x ≠ 0) := by
  sorry

end smallest_positive_root_of_f_l147_14753


namespace inequality_proof_l147_14790

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  Real.sqrt (3 * x^2 + x * y) + Real.sqrt (3 * y^2 + y * z) + Real.sqrt (3 * z^2 + z * x) ≤ 2 * (x + y + z) := by
  sorry

end inequality_proof_l147_14790


namespace sum_of_fractions_equals_one_l147_14708

theorem sum_of_fractions_equals_one
  (a b c x y z : ℝ)
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 50 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 50) = 1 := by
  sorry

end sum_of_fractions_equals_one_l147_14708


namespace twenty_three_in_base_two_l147_14773

theorem twenty_three_in_base_two : 
  (23 : ℕ) = 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 := by
  sorry

end twenty_three_in_base_two_l147_14773


namespace polynomial_expansion_properties_l147_14791

theorem polynomial_expansion_properties (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x : ℝ, (1 + 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  a₂ = 24 ∧ a + a₁ + a₂ + a₃ + a₄ = 81 := by
  sorry

end polynomial_expansion_properties_l147_14791


namespace calvins_weight_after_training_l147_14776

/-- Calculates the final weight after a period of constant weight loss -/
def final_weight (initial_weight : ℕ) (weight_loss_per_month : ℕ) (months : ℕ) : ℕ :=
  initial_weight - weight_loss_per_month * months

/-- Theorem stating that Calvin's weight after one year of training is 154 pounds -/
theorem calvins_weight_after_training :
  final_weight 250 8 12 = 154 := by
  sorry

end calvins_weight_after_training_l147_14776


namespace problem_statement_l147_14795

theorem problem_statement : (2222 - 2002)^2 / 144 = 3025 / 9 := by sorry

end problem_statement_l147_14795


namespace point_in_second_quadrant_l147_14715

theorem point_in_second_quadrant (A B C : ℝ) : 
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  A + B + C = π →    -- A, B, C are angles of a triangle
  Real.cos B - Real.sin A < 0 ∧ Real.sin B - Real.cos A > 0 := by
sorry

end point_in_second_quadrant_l147_14715


namespace f_value_at_11pi_over_6_l147_14738

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, (is_periodic f q ∧ q > 0) → p ≤ q

theorem f_value_at_11pi_over_6 (f : ℝ → ℝ) :
  is_odd f →
  smallest_positive_period f π →
  (∀ x ∈ Set.Ioo 0 (π/2), f x = 2 * Real.sin x) →
  f (11*π/6) = -1 := by sorry

end f_value_at_11pi_over_6_l147_14738


namespace find_unknown_numbers_l147_14769

/-- Given four real numbers A, B, C, and D satisfying certain conditions,
    prove that they have specific values. -/
theorem find_unknown_numbers (A B C D : ℝ) 
    (h1 : 0.05 * A = 0.20 * 650 + 0.10 * B)
    (h2 : A + B = 4000)
    (h3 : C = 2 * B)
    (h4 : A + B + C = 0.40 * D) :
    A = 3533.3333333333335 ∧ 
    B = 466.6666666666667 ∧ 
    C = 933.3333333333334 ∧ 
    D = 12333.333333333334 := by
  sorry

end find_unknown_numbers_l147_14769


namespace book_arrangement_theorem_l147_14780

/-- The number of ways to arrange two types of indistinguishable objects in a row -/
def arrange_books (n m : ℕ) : ℕ := Nat.choose (n + m) n

/-- Theorem: Arranging 5 and 6 indistinguishable objects in 11 positions yields 462 ways -/
theorem book_arrangement_theorem :
  arrange_books 5 6 = 462 := by
  sorry

end book_arrangement_theorem_l147_14780


namespace count_perfect_square_factors_l147_14751

/-- The number of perfect square factors of 345600 -/
def perfectSquareFactors : ℕ := 16

/-- The prime factorization of 345600 -/
def n : ℕ := 2^6 * 3^3 * 5^2

/-- A function that counts the number of perfect square factors of n -/
def countPerfectSquareFactors (n : ℕ) : ℕ := sorry

theorem count_perfect_square_factors :
  countPerfectSquareFactors n = perfectSquareFactors := by sorry

end count_perfect_square_factors_l147_14751


namespace cathys_wallet_theorem_l147_14723

/-- Calculates the remaining money in Cathy's wallet after receiving money from parents, buying a book, and saving some money. -/
def cathys_remaining_money (initial_amount dad_contribution book_cost savings_rate : ℚ) : ℚ :=
  let mom_contribution := 2 * dad_contribution
  let total_received := initial_amount + dad_contribution + mom_contribution
  let after_book_purchase := total_received - book_cost
  let savings_amount := savings_rate * after_book_purchase
  after_book_purchase - savings_amount

/-- Theorem stating that Cathy's remaining money is $57.60 given the initial conditions. -/
theorem cathys_wallet_theorem :
  cathys_remaining_money 12 25 15 (1/5) = 288/5 := by sorry

end cathys_wallet_theorem_l147_14723


namespace chess_tournament_games_l147_14720

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 15 players where each player plays every other player once, 
    the total number of games played is 105. -/
theorem chess_tournament_games : num_games 15 = 105 := by
  sorry

end chess_tournament_games_l147_14720


namespace prob_red_ball_one_third_l147_14736

/-- A bag containing red and yellow balls -/
structure Bag where
  red_balls : ℕ
  yellow_balls : ℕ

/-- The probability of drawing a red ball from the bag -/
def prob_red_ball (bag : Bag) : ℚ :=
  bag.red_balls / (bag.red_balls + bag.yellow_balls)

/-- The theorem stating the probability of drawing a red ball -/
theorem prob_red_ball_one_third (bag : Bag) 
  (h1 : bag.red_balls = 1) 
  (h2 : bag.yellow_balls = 2) : 
  prob_red_ball bag = 1/3 := by
  sorry

#check prob_red_ball_one_third

end prob_red_ball_one_third_l147_14736


namespace cube_volume_surface_area_l147_14725

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 0 := by
sorry

end cube_volume_surface_area_l147_14725


namespace time_left_before_movie_l147_14732

def movie_time_minutes : ℕ := 2 * 60

def homework_time : ℕ := 30

def room_cleaning_time : ℕ := homework_time / 2

def dog_walking_time : ℕ := homework_time + 5

def trash_taking_time : ℕ := homework_time / 6

def total_chore_time : ℕ := homework_time + room_cleaning_time + dog_walking_time + trash_taking_time

theorem time_left_before_movie : movie_time_minutes - total_chore_time = 35 := by
  sorry

end time_left_before_movie_l147_14732


namespace system_solution_l147_14737

-- Define the system of equations
def equation1 (x y : ℚ) : Prop := 2 * x - 3 * y = 5
def equation2 (x y : ℚ) : Prop := 4 * x + y = 9

-- Define the solution
def solution : ℚ × ℚ := (16/7, -1/7)

-- Theorem statement
theorem system_solution :
  let (x, y) := solution
  equation1 x y ∧ equation2 x y := by sorry

end system_solution_l147_14737


namespace solution_pairs_l147_14749

/-- The type of pairs of positive integers satisfying the divisibility condition -/
def SolutionPairs : Type := 
  {p : Nat × Nat // p.1 > 0 ∧ p.2 > 0 ∧ (2^(2^p.1) + 1) * (2^(2^p.2) + 1) % (p.1 * p.2) = 0}

/-- The theorem stating the solution pairs -/
theorem solution_pairs : 
  {p : SolutionPairs | p.val = (1, 1) ∨ p.val = (1, 5) ∨ p.val = (5, 1)} = 
  {p : SolutionPairs | true} := by sorry

end solution_pairs_l147_14749


namespace algebraic_identity_l147_14742

theorem algebraic_identity (a b : ℝ) : a * b - 2 * (a * b) = -(a * b) := by
  sorry

end algebraic_identity_l147_14742


namespace john_yearly_music_cost_l147_14718

/-- Calculates the yearly cost of music for John given his buying habits --/
theorem john_yearly_music_cost
  (hours_per_month : ℕ)
  (song_length_minutes : ℕ)
  (song_cost_cents : ℕ)
  (h1 : hours_per_month = 20)
  (h2 : song_length_minutes = 3)
  (h3 : song_cost_cents = 50) :
  (hours_per_month * 60 / song_length_minutes) * song_cost_cents * 12 = 240000 :=
by sorry

end john_yearly_music_cost_l147_14718


namespace parallel_planes_sum_l147_14709

/-- Given two planes α and β with normal vectors (x, 1, -2) and (-1, y, 1/2) respectively,
    if α is parallel to β, then x + y = 15/4 -/
theorem parallel_planes_sum (x y : ℝ) : 
  let n₁ : Fin 3 → ℝ := ![x, 1, -2]
  let n₂ : Fin 3 → ℝ := ![-1, y, 1/2]
  (∃ (k : ℝ), ∀ i, n₁ i = k * n₂ i) →
  x + y = 15/4 := by
sorry

end parallel_planes_sum_l147_14709


namespace special_triangle_perimeter_l147_14750

/-- A triangle with sides that are consecutive natural numbers and largest angle twice the smallest -/
structure SpecialTriangle where
  n : ℕ
  side1 : ℕ := n - 1
  side2 : ℕ := n
  side3 : ℕ := n + 1
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angle_sum : angleA + angleB + angleC = π
  angle_relation : angleC = 2 * angleA
  law_of_sines : (n - 1) / Real.sin angleA = n / Real.sin angleB
  law_of_cosines : (n - 1)^2 = (n + 1)^2 + n^2 - 2 * (n + 1) * n * Real.cos angleC

/-- The perimeter of the special triangle is 15 -/
theorem special_triangle_perimeter (t : SpecialTriangle) : t.side1 + t.side2 + t.side3 = 15 := by
  sorry

end special_triangle_perimeter_l147_14750


namespace quadratic_real_roots_range_l147_14730

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 3 * x - 9/4 = 0) ↔ (k > -1 ∨ k < -1) ∧ k ≠ 0 :=
sorry

end quadratic_real_roots_range_l147_14730


namespace a_monotonically_decreasing_iff_t_lt_3_l147_14779

/-- The sequence a_n defined as -n^2 + tn for positive integers n and constant t -/
def a (n : ℕ+) (t : ℝ) : ℝ := -n.val^2 + t * n.val

/-- A sequence is monotonically decreasing if each term is less than the previous term -/
def monotonically_decreasing (s : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, s (n + 1) < s n

/-- The main theorem: the sequence a_n is monotonically decreasing iff t < 3 -/
theorem a_monotonically_decreasing_iff_t_lt_3 (t : ℝ) :
  monotonically_decreasing (a · t) ↔ t < 3 := by
  sorry

end a_monotonically_decreasing_iff_t_lt_3_l147_14779


namespace zero_in_P_and_two_not_in_P_l147_14763

-- Define the set P
def P : Set Int := sorry

-- Define the properties of P
axiom P_contains_positive : ∃ x : Int, x > 0 ∧ x ∈ P
axiom P_contains_negative : ∃ x : Int, x < 0 ∧ x ∈ P
axiom P_contains_odd : ∃ x : Int, x % 2 ≠ 0 ∧ x ∈ P
axiom P_contains_even : ∃ x : Int, x % 2 = 0 ∧ x ∈ P
axiom P_not_contains_neg_one : -1 ∉ P
axiom P_closed_under_addition : ∀ x y : Int, x ∈ P → y ∈ P → (x + y) ∈ P

-- Theorem to prove
theorem zero_in_P_and_two_not_in_P : 0 ∈ P ∧ 2 ∉ P := by
  sorry

end zero_in_P_and_two_not_in_P_l147_14763


namespace house_number_theorem_l147_14766

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ := (n / 100) + ((n / 10) % 10) + (n % 10)

def all_digits_same (n : ℕ) : Prop :=
  (n / 100) = ((n / 10) % 10) ∧ ((n / 10) % 10) = (n % 10)

def two_digits_same (n : ℕ) : Prop :=
  (n / 100 = (n / 10) % 10) ∨ ((n / 10) % 10 = n % 10) ∨ (n / 100 = n % 10)

def all_digits_different (n : ℕ) : Prop :=
  (n / 100) ≠ ((n / 10) % 10) ∧ ((n / 10) % 10) ≠ (n % 10) ∧ (n / 100) ≠ (n % 10)

theorem house_number_theorem :
  (∃! n : ℕ, is_three_digit n ∧ digit_sum n = 24 ∧ all_digits_same n) ∧
  (∃ l : List ℕ, l.length = 3 ∧ ∀ n ∈ l, is_three_digit n ∧ digit_sum n = 24 ∧ two_digits_same n) ∧
  (∃ l : List ℕ, l.length = 6 ∧ ∀ n ∈ l, is_three_digit n ∧ digit_sum n = 24 ∧ all_digits_different n) :=
sorry

end house_number_theorem_l147_14766


namespace bounds_per_meter_proof_l147_14797

/-- Represents the number of bounds in one meter -/
def bounds_per_meter : ℚ :=
  21 / 100

/-- The number of leaps that equal 3 bounds -/
def leaps_to_bounds : ℕ := 4

/-- The number of bounds that equal 4 leaps -/
def bounds_to_leaps : ℕ := 3

/-- The number of strides that equal 2 leaps -/
def strides_to_leaps : ℕ := 5

/-- The number of leaps that equal 5 strides -/
def leaps_to_strides : ℕ := 2

/-- The number of strides that equal 10 meters -/
def strides_to_meters : ℕ := 7

/-- The number of meters that equal 7 strides -/
def meters_to_strides : ℕ := 10

theorem bounds_per_meter_proof :
  bounds_per_meter = 21 / 100 :=
by sorry

end bounds_per_meter_proof_l147_14797


namespace difference_of_sum_and_product_l147_14734

theorem difference_of_sum_and_product (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (prod_eq : x * y = 221) : 
  |x - y| = 4 := by sorry

end difference_of_sum_and_product_l147_14734


namespace gcd_of_B_is_two_l147_14710

def B : Set ℕ := {n | ∃ x : ℕ, x > 0 ∧ n = (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end gcd_of_B_is_two_l147_14710


namespace tangent_line_circle_range_l147_14731

theorem tangent_line_circle_range (m n : ℝ) : 
  (∃ (x y : ℝ), (m + 1) * x + (n + 1) * y - 2 = 0 ∧ 
   (x - 1)^2 + (y - 1)^2 = 1 ∧ 
   ∀ (x' y' : ℝ), (m + 1) * x' + (n + 1) * y' - 2 = 0 → (x' - 1)^2 + (y' - 1)^2 ≥ 1) →
  m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

end tangent_line_circle_range_l147_14731


namespace bus_ticket_cost_l147_14772

theorem bus_ticket_cost 
  (total_tickets : ℕ)
  (senior_ticket_cost : ℕ)
  (total_sales : ℕ)
  (senior_tickets_sold : ℕ)
  (h1 : total_tickets = 65)
  (h2 : senior_ticket_cost = 10)
  (h3 : total_sales = 855)
  (h4 : senior_tickets_sold = 24) :
  (total_sales - senior_tickets_sold * senior_ticket_cost) / (total_tickets - senior_tickets_sold) = 15 :=
by sorry

end bus_ticket_cost_l147_14772


namespace box_side_length_l147_14789

/-- Proves that the length of one side of a cubic box can be calculated given the total volume,
    total cost, and cost per box. -/
theorem box_side_length 
  (cost_per_box : ℝ) 
  (total_volume : ℝ) 
  (total_cost : ℝ) 
  (cost_per_box_positive : cost_per_box > 0)
  (total_volume_positive : total_volume > 0)
  (total_cost_positive : total_cost > 0) :
  ∃ (side_length : ℝ), 
    side_length = (total_volume / (total_cost / cost_per_box)) ^ (1/3) :=
by sorry

end box_side_length_l147_14789


namespace inequality_solution_set_l147_14761

theorem inequality_solution_set (x : ℝ) : (2 * x - 1) / (3 * x + 1) > 1 ↔ -2 < x ∧ x < 1/3 := by
  sorry

end inequality_solution_set_l147_14761


namespace rectangle_composition_l147_14757

/-- The side length of the middle square in a specific rectangular arrangement -/
def square_side_length : ℝ := by sorry

theorem rectangle_composition (total_width total_height : ℝ) 
  (h_width : total_width = 3500)
  (h_height : total_height = 2100)
  (h_composition : ∃ (r : ℝ), 2 * r + square_side_length = total_height ∧ 
                               (square_side_length + 100) + square_side_length + (square_side_length + 200) = total_width) :
  square_side_length = 1066.67 := by sorry

end rectangle_composition_l147_14757


namespace evaluate_expression_l147_14782

theorem evaluate_expression : (528 : ℤ) * 528 - (527 * 529) = 1 := by
  sorry

end evaluate_expression_l147_14782


namespace simplify_fraction_l147_14775

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (45 * b^3) = 2 / 3 := by
  sorry

end simplify_fraction_l147_14775


namespace polynomial_factor_l147_14752

/-- The polynomial with parameters a and b -/
def P (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^3 + 48 * x^2 - 24 * x + 4

/-- The factor of the polynomial -/
def F (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 1

/-- Theorem stating that the polynomial P has the factor F when a = -16 and b = -36 -/
theorem polynomial_factor (x : ℝ) : ∃ (Q : ℝ → ℝ), P (-16) (-36) x = F x * Q x := by
  sorry

end polynomial_factor_l147_14752


namespace largest_sample_number_l147_14756

/-- Systematic sampling from a set of numbered items -/
def systematic_sample (total : ℕ) (first : ℕ) (second : ℕ) : ℕ := 
  let interval := second - first
  let sample_size := total / interval
  first + interval * (sample_size - 1)

/-- The largest number in a systematic sample from 500 items -/
theorem largest_sample_number : 
  systematic_sample 500 7 32 = 482 := by
  sorry

end largest_sample_number_l147_14756


namespace journey_distance_is_420_l147_14707

/-- Represents the journey details -/
structure Journey where
  urban_speed : ℝ
  highway_speed : ℝ
  urban_time : ℝ
  highway_time : ℝ

/-- Calculates the total distance of the journey -/
def total_distance (j : Journey) : ℝ :=
  j.urban_speed * j.urban_time + j.highway_speed * j.highway_time

/-- Theorem stating that the journey distance is 420 km -/
theorem journey_distance_is_420 (j : Journey) 
  (h1 : j.urban_speed = 55)
  (h2 : j.highway_speed = 85)
  (h3 : j.urban_time = 3)
  (h4 : j.highway_time = 3) :
  total_distance j = 420 := by
  sorry

#eval total_distance { urban_speed := 55, highway_speed := 85, urban_time := 3, highway_time := 3 }

end journey_distance_is_420_l147_14707


namespace complex_symmetry_product_l147_14714

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  (z₁.im = -z₂.im) → (z₁.re = z₂.re) → (z₁ ≠ 1 + I) → z₁ * z₂ = 2 := by
  sorry

end complex_symmetry_product_l147_14714


namespace product_divisible_by_5184_l147_14712

theorem product_divisible_by_5184 (k m : ℕ) : 
  5184 ∣ ((k^3 - 1) * k^3 * (k^3 + 1) * (m^3 - 1) * m^3 * (m^3 + 1)) := by
sorry

end product_divisible_by_5184_l147_14712


namespace females_in_coach_class_l147_14781

theorem females_in_coach_class 
  (total_passengers : ℕ) 
  (female_percentage : ℚ) 
  (first_class_percentage : ℚ) 
  (male_first_class_fraction : ℚ) 
  (h1 : total_passengers = 120)
  (h2 : female_percentage = 30 / 100)
  (h3 : first_class_percentage = 10 / 100)
  (h4 : male_first_class_fraction = 1 / 3) :
  ↑((total_passengers : ℚ) * female_percentage - 
    (total_passengers : ℚ) * first_class_percentage * (1 - male_first_class_fraction)) = 28 := by
  sorry

end females_in_coach_class_l147_14781


namespace inscribed_square_in_acute_triangle_l147_14770

/-- A triangle is acute-angled if all its angles are less than 90 degrees -/
def IsAcuteAngledTriangle (A B C : Point) : Prop := sorry

/-- A square is inscribed in a triangle if all its vertices lie on the sides of the triangle -/
def IsInscribedSquare (K L M N : Point) (A B C : Point) : Prop := sorry

/-- Two points lie on the same side of a triangle -/
def LieOnSameSide (P Q : Point) (A B C : Point) : Prop := sorry

theorem inscribed_square_in_acute_triangle 
  (A B C : Point) (h : IsAcuteAngledTriangle A B C) :
  ∃ (K L M N : Point), 
    IsInscribedSquare K L M N A B C ∧ 
    LieOnSameSide L M A B C ∧
    ((LieOnSameSide K N A B C ∧ ¬LieOnSameSide K N B C A) ∨
     (LieOnSameSide K N B C A ∧ ¬LieOnSameSide K N A B C)) :=
sorry

end inscribed_square_in_acute_triangle_l147_14770


namespace tangent_line_point_on_circle_l147_14741

/-- Given a circle C defined by x^2 + y^2 = 1 and a line L defined by ax + by = 1 
    that is tangent to C, prove that the point (a, b) lies on C. -/
theorem tangent_line_point_on_circle (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (a*x + b*y = 1 → x^2 + y^2 > 1 ∨ a*x + b*y > 1)) → 
  a^2 + b^2 = 1 := by
  sorry

#check tangent_line_point_on_circle

end tangent_line_point_on_circle_l147_14741


namespace circle_condition_l147_14767

-- Define the equation
def circle_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0

-- Theorem statement
theorem circle_condition (a : ℝ) :
  (∃ h k r, ∀ x y, circle_equation a x y ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ a = 2 :=
sorry

end circle_condition_l147_14767


namespace postal_stamp_problem_l147_14788

theorem postal_stamp_problem :
  ∀ (x : ℕ),
  (75 : ℕ) = 40 + (75 - 40) →
  (480 : ℕ) = 40 * 5 + (75 - 40) * x →
  x = 8 := by
sorry

end postal_stamp_problem_l147_14788


namespace circumradius_eq_one_l147_14701

/-- Three unit circles passing through a common point -/
structure ThreeIntersectingCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  center3 : ℝ × ℝ
  commonPoint : ℝ × ℝ
  radius : ℝ
  radius_eq_one : radius = 1
  passes_through_common : 
    dist center1 commonPoint = radius ∧
    dist center2 commonPoint = radius ∧
    dist center3 commonPoint = radius

/-- The three intersection points forming triangle ABC -/
def intersectionPoints (c : ThreeIntersectingCircles) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

/-- The circumcenter of triangle ABC -/
def circumcenter (points : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

/-- The circumradius of triangle ABC -/
def circumradius (c : ThreeIntersectingCircles) : ℝ :=
  let points := intersectionPoints c
  dist (circumcenter points) points.1

/-- Theorem: The circumradius of triangle ABC is equal to 1 -/
theorem circumradius_eq_one (c : ThreeIntersectingCircles) :
  circumradius c = 1 :=
sorry

end circumradius_eq_one_l147_14701


namespace solve_equation_l147_14747

theorem solve_equation (x : ℝ) : 2 - 2 / (1 - x) = 2 / (1 - x) → x = -2 := by
  sorry

end solve_equation_l147_14747


namespace complement_of_P_l147_14724

-- Define the universal set R as the set of real numbers
def R : Set ℝ := Set.univ

-- Define set P
def P : Set ℝ := {x : ℝ | x ≥ 1}

-- State the theorem
theorem complement_of_P : 
  Set.compl P = {x : ℝ | x < 1} :=
by
  sorry

end complement_of_P_l147_14724


namespace stating_clock_hands_overlap_at_316_l147_14798

/-- Represents the number of degrees the hour hand moves in one minute -/
def hourHandDegPerMin : ℝ := 0.5

/-- Represents the number of degrees the minute hand moves in one minute -/
def minuteHandDegPerMin : ℝ := 6

/-- Represents the number of degrees between the hour and minute hands at 3:00 -/
def initialAngle : ℝ := 90

/-- 
Theorem stating that the hour and minute hands of a clock overlap 16 minutes after 3:00
-/
theorem clock_hands_overlap_at_316 :
  ∃ (x : ℝ), x > 0 ∧ x < 60 ∧ 
  minuteHandDegPerMin * x - hourHandDegPerMin * x = initialAngle ∧
  x = 16 := by
  sorry

end stating_clock_hands_overlap_at_316_l147_14798


namespace inequality_proof_l147_14755

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) :
  (1 / (2 - a)) + (1 / (2 - b)) + (1 / (2 - c)) ≥ 3 := by
  sorry

end inequality_proof_l147_14755


namespace picture_area_l147_14768

theorem picture_area (x y : ℕ) (h1 : x > 1) (h2 : y > 1) (h3 : (3 * x + 3) * (y + 2) = 110) : x * y = 28 := by
  sorry

end picture_area_l147_14768


namespace minimum_value_and_max_when_half_l147_14759

noncomputable def f (a x : ℝ) : ℝ := 1 - 2*a - 2*a*Real.cos x - 2*(Real.sin x)^2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a ≤ 2 then -a^2/2 - 2*a - 1
  else 1 - 4*a

theorem minimum_value_and_max_when_half (a : ℝ) :
  (∀ x, f a x ≥ g a) ∧
  (g a = 1/2 → a = -1 ∧ ∃ x, f (-1) x = 5 ∧ ∀ y, f (-1) y ≤ 5) :=
sorry

end minimum_value_and_max_when_half_l147_14759


namespace mod_fifteen_equivalence_l147_14748

theorem mod_fifteen_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 7615 [ZMOD 15] ∧ n = 10 := by
  sorry

end mod_fifteen_equivalence_l147_14748


namespace remaining_balance_calculation_l147_14783

def initial_balance : ℝ := 50
def coffee_expense : ℝ := 10
def tumbler_expense : ℝ := 30

theorem remaining_balance_calculation :
  initial_balance - (coffee_expense + tumbler_expense) = 10 := by
  sorry

end remaining_balance_calculation_l147_14783


namespace mileage_difference_l147_14745

/-- The difference between advertised and actual mileage -/
theorem mileage_difference (advertised_mpg : ℝ) (tank_capacity : ℝ) (miles_driven : ℝ) :
  advertised_mpg = 35 →
  tank_capacity = 12 →
  miles_driven = 372 →
  advertised_mpg - (miles_driven / tank_capacity) = 4 := by
  sorry

end mileage_difference_l147_14745


namespace continuity_definition_relation_l147_14764

-- Define a real-valued function
variable (f : ℝ → ℝ)
-- Define a point x₀
variable (x₀ : ℝ)

-- Define what it means for f to be defined at x₀
def is_defined_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ y : ℝ, f x₀ = y

-- State the theorem
theorem continuity_definition_relation :
  (ContinuousAt f x₀ → is_defined_at f x₀) ∧
  ¬(is_defined_at f x₀ → ContinuousAt f x₀) :=
sorry

end continuity_definition_relation_l147_14764


namespace intersection_of_A_and_B_l147_14704

open Set

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l147_14704


namespace steven_jill_peach_difference_l147_14744

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 19

/-- The number of peaches Jill has -/
def jill_peaches : ℕ := 6

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - 18

/-- Theorem: Steven has 13 more peaches than Jill -/
theorem steven_jill_peach_difference : steven_peaches - jill_peaches = 13 := by
  sorry

end steven_jill_peach_difference_l147_14744


namespace harvest_duration_proof_l147_14758

/-- Calculates the number of weeks the harvest lasted. -/
def harvest_duration (weekly_earnings : ℕ) (total_earnings : ℕ) : ℕ :=
  total_earnings / weekly_earnings

/-- Proves that the harvest lasted 89 weeks given the conditions. -/
theorem harvest_duration_proof (weekly_earnings total_earnings : ℕ) 
  (h1 : weekly_earnings = 2)
  (h2 : total_earnings = 178) :
  harvest_duration weekly_earnings total_earnings = 89 := by
  sorry

end harvest_duration_proof_l147_14758


namespace stratified_sample_size_l147_14716

/-- Represents the total number of students in the school -/
def total_students : ℕ := 600 + 500 + 400

/-- Represents the number of students in the first grade -/
def first_grade_students : ℕ := 600

/-- Represents the number of first-grade students in the sample -/
def first_grade_sample : ℕ := 30

/-- Theorem stating that the total sample size is 75 given the conditions -/
theorem stratified_sample_size :
  ∃ (n : ℕ),
    n * first_grade_students = total_students * first_grade_sample ∧
    n = 75 := by
  sorry

end stratified_sample_size_l147_14716


namespace adults_average_age_l147_14765

def robotics_camp_problem (total_members : ℕ) (overall_average_age : ℝ)
  (num_girls num_boys num_adults : ℕ) (girls_average_age boys_average_age : ℝ) : Prop :=
  total_members = 50 ∧
  overall_average_age = 20 ∧
  num_girls = 25 ∧
  num_boys = 18 ∧
  num_adults = 7 ∧
  girls_average_age = 18 ∧
  boys_average_age = 19 ∧
  (total_members : ℝ) * overall_average_age =
    (num_girls : ℝ) * girls_average_age +
    (num_boys : ℝ) * boys_average_age +
    (num_adults : ℝ) * ((1000 - 450 - 342) / 7)

theorem adults_average_age
  (total_members : ℕ) (overall_average_age : ℝ)
  (num_girls num_boys num_adults : ℕ) (girls_average_age boys_average_age : ℝ)
  (h : robotics_camp_problem total_members overall_average_age
    num_girls num_boys num_adults girls_average_age boys_average_age) :
  (1000 - 450 - 342) / 7 = (total_members * overall_average_age -
    num_girls * girls_average_age - num_boys * boys_average_age) / num_adults :=
by sorry

end adults_average_age_l147_14765


namespace wallet_problem_l147_14785

/-- The number of quarters in the wallet -/
def num_quarters : ℕ := 15

/-- The number of dimes in the wallet -/
def num_dimes : ℕ := 25

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of dimes that equal the value of the quarters -/
def n : ℕ := 38

theorem wallet_problem :
  (num_quarters * quarter_value : ℕ) = n * dime_value :=
by sorry

end wallet_problem_l147_14785


namespace joanne_earnings_l147_14777

/-- Joanne's work schedule and earnings calculation -/
theorem joanne_earnings :
  let main_job_hours : ℕ := 8
  let main_job_rate : ℚ := 16
  let part_time_hours : ℕ := 2
  let part_time_rate : ℚ := 27/2  -- $13.50 represented as a fraction
  let days_worked : ℕ := 5
  
  let main_job_daily := main_job_hours * main_job_rate
  let part_time_daily := part_time_hours * part_time_rate
  let total_daily := main_job_daily + part_time_daily
  let total_weekly := total_daily * days_worked
  
  total_weekly = 775
:= by sorry


end joanne_earnings_l147_14777


namespace average_fish_is_75_l147_14705

/-- The number of fish in Boast Pool -/
def boast_pool : ℕ := 75

/-- The number of fish in Onum Lake -/
def onum_lake : ℕ := boast_pool + 25

/-- The number of fish in Riddle Pond -/
def riddle_pond : ℕ := onum_lake / 2

/-- The total number of fish in all three bodies of water -/
def total_fish : ℕ := boast_pool + onum_lake + riddle_pond

/-- The number of bodies of water -/
def num_bodies : ℕ := 3

/-- Theorem stating that the average number of fish in all three bodies of water is 75 -/
theorem average_fish_is_75 : total_fish / num_bodies = 75 := by
  sorry

end average_fish_is_75_l147_14705


namespace inverse_f_at_negative_31_96_l147_14727

noncomputable def f (x : ℝ) : ℝ := (x^5 - 1) / 3

theorem inverse_f_at_negative_31_96 : f⁻¹ (-31/96) = 1/2 := by
  sorry

end inverse_f_at_negative_31_96_l147_14727


namespace wooden_strip_triangle_l147_14796

theorem wooden_strip_triangle (x : ℝ) : 
  (0 < x ∧ x < 5 ∧ 
   x + x > 10 - 2*x ∧
   10 - 2*x > 0) ↔ 
  (2.5 < x ∧ x < 5) :=
sorry

end wooden_strip_triangle_l147_14796


namespace total_cost_is_correct_l147_14740

def type_a_cost : ℚ := 9
def type_a_quantity : ℕ := 4
def type_b_extra_cost : ℚ := 5
def type_b_quantity : ℕ := 2
def clay_pot_extra_cost : ℚ := 20
def soil_cost_reduction : ℚ := 2
def fertilizer_percentage : ℚ := 1.5
def gardening_tools_percentage : ℚ := 0.75

def total_cost : ℚ :=
  type_a_cost * type_a_quantity +
  (type_a_cost + type_b_extra_cost) * type_b_quantity +
  (type_a_cost + clay_pot_extra_cost) +
  (type_a_cost - soil_cost_reduction) +
  (type_a_cost * fertilizer_percentage) +
  ((type_a_cost + clay_pot_extra_cost) * gardening_tools_percentage)

theorem total_cost_is_correct : total_cost = 135.25 := by
  sorry

end total_cost_is_correct_l147_14740


namespace ginas_expenses_theorem_l147_14739

/-- Calculates Gina's total college expenses for the year --/
def ginasCollegeExpenses : ℕ :=
  let totalCredits : ℕ := 18
  let regularCredits : ℕ := 12
  let labCredits : ℕ := 6
  let regularCreditCost : ℕ := 450
  let labCreditCost : ℕ := 550
  let textbookCount : ℕ := 3
  let textbookCost : ℕ := 150
  let onlineResourceCount : ℕ := 4
  let onlineResourceCost : ℕ := 95
  let facilitiesFee : ℕ := 200
  let labFeePerCredit : ℕ := 75

  regularCredits * regularCreditCost +
  labCredits * labCreditCost +
  textbookCount * textbookCost +
  onlineResourceCount * onlineResourceCost +
  facilitiesFee +
  labCredits * labFeePerCredit

theorem ginas_expenses_theorem : ginasCollegeExpenses = 10180 := by
  sorry

end ginas_expenses_theorem_l147_14739


namespace sum_of_roots_equals_one_l147_14726

theorem sum_of_roots_equals_one :
  let f : ℝ → ℝ := λ x ↦ (x + 3) * (x - 4) - 20
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 1 := by
  sorry

end sum_of_roots_equals_one_l147_14726


namespace linear_function_properties_l147_14719

/-- Linear function definition -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (2*m + 1)*x + m - 2

theorem linear_function_properties :
  ∀ m : ℝ,
  (∀ x, linear_function m x = 0 → x = 0) → m = 2 ∧
  (linear_function m 0 = -3) → m = -1 ∧
  (∀ x, ∃ k, linear_function m x = x + k) → m = 0 ∧
  (∀ x, x < 0 → linear_function m x > 0) → -1/2 < m ∧ m < 2 :=
by sorry

end linear_function_properties_l147_14719


namespace initial_brownies_count_l147_14787

/-- The number of brownies initially made by Mother -/
def initial_brownies : ℕ := sorry

/-- The number of brownies eaten by Father -/
def father_eaten : ℕ := 8

/-- The number of brownies eaten by Mooney -/
def mooney_eaten : ℕ := 4

/-- The number of new brownies added the next morning -/
def new_brownies : ℕ := 24

/-- The total number of brownies after adding the new ones -/
def total_brownies : ℕ := 36

theorem initial_brownies_count : initial_brownies = 24 := by
  sorry

end initial_brownies_count_l147_14787


namespace max_students_is_25_l147_14778

/-- Represents the field trip problem with given conditions --/
structure FieldTrip where
  bus_rental : ℕ
  bus_capacity : ℕ
  admission_cost : ℕ
  total_budget : ℕ

/-- Calculates the maximum number of students that can go on the field trip --/
def max_students (trip : FieldTrip) : ℕ :=
  min
    ((trip.total_budget - trip.bus_rental) / trip.admission_cost)
    trip.bus_capacity

/-- Theorem stating that the maximum number of students for the given conditions is 25 --/
theorem max_students_is_25 :
  let trip : FieldTrip := {
    bus_rental := 100,
    bus_capacity := 25,
    admission_cost := 10,
    total_budget := 350
  }
  max_students trip = 25 := by
  sorry


end max_students_is_25_l147_14778


namespace age_difference_l147_14711

theorem age_difference (patrick michael monica : ℕ) : 
  patrick * 5 = michael * 3 →
  michael * 5 = monica * 3 →
  patrick + michael + monica = 147 →
  monica - patrick = 48 :=
by sorry

end age_difference_l147_14711


namespace similar_squares_side_length_l147_14735

/-- Given two similar squares with an area ratio of 1:9 and the smaller square's side length of 5 cm,
    prove that the larger square's side length is 15 cm. -/
theorem similar_squares_side_length (small_side : ℝ) (large_side : ℝ) : 
  small_side = 5 →  -- The side length of the smaller square is 5 cm
  (large_side / small_side)^2 = 9 →  -- The ratio of their areas is 1:9
  large_side = 15 :=  -- The side length of the larger square is 15 cm
by
  sorry

end similar_squares_side_length_l147_14735


namespace square_difference_153_147_l147_14746

theorem square_difference_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end square_difference_153_147_l147_14746


namespace geometric_series_relation_l147_14786

/-- Given real numbers c and d satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 3/4. -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (c / d) / (1 - 1 / d) = 6) :
    (c / (c + 2 * d)) / (1 - 1 / (c + 2 * d)) = 3 / 4 := by
  sorry

end geometric_series_relation_l147_14786


namespace area_of_triangle_AEC_l147_14717

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the properties of the rectangle and point E
def is_rectangle (A B C D : ℝ × ℝ) : Prop := sorry

def on_segment (E C D : ℝ × ℝ) : Prop := sorry

def segment_ratio (D E C : ℝ × ℝ) (r : ℚ) : Prop := sorry

def triangle_area (A D E : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_AEC 
  (h_rectangle : is_rectangle A B C D)
  (h_on_segment : on_segment E C D)
  (h_ratio : segment_ratio D E C (3/2))
  (h_area_ADE : triangle_area A D E = 27) :
  triangle_area A E C = 18 := by sorry

end area_of_triangle_AEC_l147_14717


namespace range_of_x_l147_14799

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) :
  x ≤ -2 ∨ x ≥ 3 := by
sorry

end range_of_x_l147_14799


namespace physics_marks_calculation_l147_14774

def english_marks : ℕ := 91
def math_marks : ℕ := 65
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def total_subjects : ℕ := 5
def average_marks : ℕ := 78

theorem physics_marks_calculation :
  let known_marks := english_marks + math_marks + chemistry_marks + biology_marks
  let total_marks := average_marks * total_subjects
  total_marks - known_marks = 82 := by
  sorry

end physics_marks_calculation_l147_14774


namespace two_colors_sufficient_l147_14754

/-- Represents a key on the ring -/
structure Key where
  position : Fin 8
  color : Bool

/-- Represents the ring of keys -/
def KeyRing : Type := Fin 8 → Key

/-- A coloring scheme is valid if it allows each key to be uniquely identified -/
def is_valid_coloring (ring : KeyRing) : Prop :=
  ∀ (i j : Fin 8), i ≠ j → 
    ∃ (k : ℕ), (ring ((i + k) % 8)).color ≠ (ring ((j + k) % 8)).color

/-- There exists a valid coloring scheme using only two colors -/
theorem two_colors_sufficient : 
  ∃ (ring : KeyRing), (∀ k, (ring k).color = true ∨ (ring k).color = false) ∧ is_valid_coloring ring := by
  sorry


end two_colors_sufficient_l147_14754


namespace rice_price_decrease_l147_14703

/-- Calculates the percentage decrease in price given the original and new quantities that can be purchased with the same amount of money. -/
def price_decrease_percentage (original_quantity : ℕ) (new_quantity : ℕ) : ℚ :=
  (1 - original_quantity / new_quantity) * 100

/-- Theorem stating that if 20 kg of rice can now buy 25 kg after a price decrease, the percentage decrease is 20%. -/
theorem rice_price_decrease : price_decrease_percentage 20 25 = 20 := by
  sorry

end rice_price_decrease_l147_14703


namespace no_prime_sum_for_10003_l147_14702

/-- A function that returns the number of ways to write a natural number as the sum of two primes -/
def count_prime_sum_representations (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that 10003 cannot be written as the sum of two primes -/
theorem no_prime_sum_for_10003 : count_prime_sum_representations 10003 = 0 := by
  sorry

end no_prime_sum_for_10003_l147_14702


namespace geometric_sequence_seventh_term_l147_14728

theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℝ),
    (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
    a 1 = 4 →                            -- First term
    a 10 = 93312 →                       -- Last term
    a 7 = 186624 :=                      -- Seventh term
by sorry

end geometric_sequence_seventh_term_l147_14728


namespace solve_parking_problem_l147_14729

def parking_problem (initial_balance : ℝ) (first_ticket_cost : ℝ) (num_full_cost_tickets : ℕ) (third_ticket_fraction : ℝ) (roommate_share : ℝ) : Prop :=
  let total_cost := first_ticket_cost * num_full_cost_tickets + first_ticket_cost * third_ticket_fraction
  let james_share := total_cost * (1 - roommate_share)
  initial_balance - james_share = 325

theorem solve_parking_problem :
  parking_problem 500 150 2 (1/3) (1/2) :=
by
  sorry

#check solve_parking_problem

end solve_parking_problem_l147_14729


namespace noemi_initial_money_l147_14743

def roulette_loss : Int := 600
def blackjack_win : Int := 400
def poker_loss : Int := 400
def baccarat_win : Int := 500
def meal_cost : Int := 200
def final_amount : Int := 1800

theorem noemi_initial_money :
  ∃ (initial_money : Int),
    initial_money = 
      roulette_loss + blackjack_win + poker_loss + baccarat_win + meal_cost + final_amount :=
by
  sorry

end noemi_initial_money_l147_14743


namespace no_solution_iff_n_eq_neg_two_l147_14760

theorem no_solution_iff_n_eq_neg_two (n : ℝ) :
  (∀ x y z : ℝ, (n * x + y + z = 2 ∧ x + n * y + z = 2 ∧ x + y + n * z = 2) → False) ↔ n = -2 := by
  sorry

end no_solution_iff_n_eq_neg_two_l147_14760


namespace wendy_full_face_time_l147_14784

/-- Calculates the total time for Wendy's "full face" routine -/
def full_face_time (num_products : ℕ) (wait_time : ℕ) (makeup_time : ℕ) : ℕ :=
  (num_products - 1) * wait_time + makeup_time

/-- Proves that Wendy's "full face" routine takes 50 minutes -/
theorem wendy_full_face_time :
  full_face_time 5 5 30 = 50 := by
  sorry

end wendy_full_face_time_l147_14784
