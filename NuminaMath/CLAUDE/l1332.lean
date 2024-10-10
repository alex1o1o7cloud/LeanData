import Mathlib

namespace floor_negative_seven_fourths_l1332_133202

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end floor_negative_seven_fourths_l1332_133202


namespace quadratic_equation_solutions_l1332_133229

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ x^2 - x
  ∃ (x₁ x₂ : ℝ), (f x₁ = 0 ∧ f x₂ = 0) ∧ x₁ = 0 ∧ x₂ = 1 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end quadratic_equation_solutions_l1332_133229


namespace pencil_difference_l1332_133215

/-- The cost of pencils Jamar bought -/
def jamar_cost : ℚ := 325 / 100

/-- The cost of pencils Sharona bought -/
def sharona_cost : ℚ := 425 / 100

/-- The minimum number of pencils Jamar bought -/
def jamar_min_pencils : ℕ := 15

/-- The cost difference between Sharona's and Jamar's purchases -/
def cost_difference : ℚ := sharona_cost - jamar_cost

/-- The theorem stating the difference in the number of pencils bought -/
theorem pencil_difference : ∃ (jamar_pencils sharona_pencils : ℕ) (price_per_pencil : ℚ), 
  jamar_pencils ≥ jamar_min_pencils ∧
  price_per_pencil > 1 / 100 ∧
  jamar_cost = jamar_pencils * price_per_pencil ∧
  sharona_cost = sharona_pencils * price_per_pencil ∧
  sharona_pencils - jamar_pencils = 5 := by
  sorry

end pencil_difference_l1332_133215


namespace largest_divisor_of_sequence_l1332_133226

theorem largest_divisor_of_sequence (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k = 105 ∧ 
  (∀ m : ℕ, m > k → ¬(m ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13))) ∧
  (k ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) := by
  sorry

end largest_divisor_of_sequence_l1332_133226


namespace cookies_per_bag_l1332_133288

/-- Given 33 cookies distributed equally among 3 bags, prove that each bag contains 11 cookies. -/
theorem cookies_per_bag :
  ∀ (total_cookies : ℕ) (num_bags : ℕ) (cookies_per_bag : ℕ),
    total_cookies = 33 →
    num_bags = 3 →
    total_cookies = num_bags * cookies_per_bag →
    cookies_per_bag = 11 := by
  sorry

end cookies_per_bag_l1332_133288


namespace map_scale_l1332_133251

theorem map_scale (map_length : ℝ) (actual_distance : ℝ) :
  (15 : ℝ) * actual_distance = 90 * map_length →
  (20 : ℝ) * actual_distance = 120 * map_length :=
by sorry

end map_scale_l1332_133251


namespace probability_white_and_black_l1332_133204

def total_balls : ℕ := 6
def red_balls : ℕ := 1
def white_balls : ℕ := 2
def black_balls : ℕ := 3
def drawn_balls : ℕ := 2

def favorable_outcomes : ℕ := white_balls * black_balls
def total_outcomes : ℕ := total_balls.choose drawn_balls

theorem probability_white_and_black :
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 5 := by sorry

end probability_white_and_black_l1332_133204


namespace same_solution_k_value_l1332_133272

theorem same_solution_k_value (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 = 0 ↔ 5 * x + 3 * k = 21) → k = 8 := by
  sorry

end same_solution_k_value_l1332_133272


namespace tangent_line_equation_l1332_133235

/-- The function f(x) = x³ - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 0)

/-- Theorem: The equation of the tangent line to f(x) at (1, 0) is 2x - y - 2 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | 2 * x - y - 2 = 0} ↔
    y - f point.1 = f' point.1 * (x - point.1) :=
by sorry

end tangent_line_equation_l1332_133235


namespace james_cheezits_consumption_l1332_133262

/-- Represents the number of bags of Cheezits James ate -/
def bags_of_cheezits : ℕ := sorry

/-- Represents the weight of each bag of Cheezits in ounces -/
def bag_weight : ℕ := 2

/-- Represents the number of calories in an ounce of Cheezits -/
def calories_per_ounce : ℕ := 150

/-- Represents the duration of James' run in minutes -/
def run_duration : ℕ := 40

/-- Represents the number of calories burned per minute during the run -/
def calories_burned_per_minute : ℕ := 12

/-- Represents the excess calories James consumed -/
def excess_calories : ℕ := 420

theorem james_cheezits_consumption :
  bags_of_cheezits * (bag_weight * calories_per_ounce) - 
  (run_duration * calories_burned_per_minute) = excess_calories ∧
  bags_of_cheezits = 3 := by sorry

end james_cheezits_consumption_l1332_133262


namespace inequalities_proof_l1332_133265

theorem inequalities_proof (a b r s : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : r > 0) (hs : s > 0) 
  (hrs : 1/r + 1/s = 1) : 
  (a^2 * b ≤ 4 * ((a + b) / 3)^3) ∧ 
  ((a^r / r) + (b^s / s) ≥ a * b) := by
sorry

end inequalities_proof_l1332_133265


namespace function_symmetry_l1332_133239

/-- Given a real-valued function f(x) = x³ + sin(x) + 1 and a real number a such that f(a) = 2,
    prove that f(-a) = 0. -/
theorem function_symmetry (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^3 + Real.sin x + 1) 
    (h2 : f a = 2) : f (-a) = 0 := by
  sorry

end function_symmetry_l1332_133239


namespace inequality_solution_set_l1332_133217

theorem inequality_solution_set : 
  {x : ℝ | x^2 + x - 2 > 0} = {x : ℝ | x < -2 ∨ x > 1} := by sorry

end inequality_solution_set_l1332_133217


namespace stock_selection_probabilities_l1332_133234

/-- The number of stocks available for purchase -/
def num_stocks : ℕ := 10

/-- The number of people buying stocks -/
def num_people : ℕ := 3

/-- The probability of all people selecting the same stock -/
def prob_all_same : ℚ := 1 / 100

/-- The probability of at least two people selecting the same stock -/
def prob_at_least_two_same : ℚ := 7 / 25

/-- Theorem stating the probabilities for the stock selection problem -/
theorem stock_selection_probabilities :
  (prob_all_same = 1 / num_stocks ^ (num_people - 1)) ∧
  (prob_at_least_two_same = 
    (1 / num_stocks ^ (num_people - 1)) + 
    (num_stocks * (num_people.choose 2) * (1 / num_stocks ^ 2) * ((num_stocks - 1) / num_stocks))) :=
by sorry

end stock_selection_probabilities_l1332_133234


namespace rectangle_width_l1332_133282

/-- Given a rectangle with length 4 times its width and area 196 square inches, 
    prove that its width is 7 inches. -/
theorem rectangle_width (w : ℝ) (h1 : w > 0) (h2 : w * (4 * w) = 196) : w = 7 := by
  sorry

end rectangle_width_l1332_133282


namespace q_equals_six_l1332_133256

/-- Represents a digit from 4 to 9 -/
def Digit := {n : ℕ // 4 ≤ n ∧ n ≤ 9}

/-- The theorem stating that Q must be 6 given the conditions -/
theorem q_equals_six 
  (P Q R S T U : Digit) 
  (unique : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
            Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
            R ≠ S ∧ R ≠ T ∧ R ≠ U ∧
            S ≠ T ∧ S ≠ U ∧
            T ≠ U)
  (sum_constraint : P.val + Q.val + S.val + 
                    T.val + U.val + R.val + 
                    P.val + T.val + S.val + 
                    R.val + Q.val + S.val + 
                    P.val + U.val = 100) : 
  Q.val = 6 := by
  sorry

end q_equals_six_l1332_133256


namespace intersection_symmetry_implies_k_minus_m_eq_four_l1332_133208

/-- The line equation y = kx + 1 -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

/-- The circle equation x² + y² + kx + my - 4 = 0 -/
def circle_equation (k m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + k*x + m*y - 4 = 0

/-- The symmetry line equation x + y - 1 = 0 -/
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- Two points (x₁, y₁) and (x₂, y₂) are symmetric with respect to the line x + y - 1 = 0 -/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 + (y₁ + y₂) / 2 - 1 = 0

theorem intersection_symmetry_implies_k_minus_m_eq_four (k m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    line_equation k x₁ y₁ ∧
    line_equation k x₂ y₂ ∧
    circle_equation k m x₁ y₁ ∧
    circle_equation k m x₂ y₂ ∧
    symmetric_points x₁ y₁ x₂ y₂) →
  k - m = 4 := by
  sorry

end intersection_symmetry_implies_k_minus_m_eq_four_l1332_133208


namespace T1_T2_T3_l1332_133201

-- Define the types for pib and maa
variable (Pib Maa : Type)

-- Define the belongs_to relation
variable (belongs_to : Maa → Pib → Prop)

-- P1: Every pib is a collection of maas
axiom P1 : ∀ p : Pib, ∃ m : Maa, belongs_to m p

-- P2: Any two distinct pibs have one and only one maa in common
axiom P2 : ∀ p1 p2 : Pib, p1 ≠ p2 → ∃! m : Maa, belongs_to m p1 ∧ belongs_to m p2

-- P3: Every maa belongs to two and only two pibs
axiom P3 : ∀ m : Maa, ∃! p1 p2 : Pib, p1 ≠ p2 ∧ belongs_to m p1 ∧ belongs_to m p2

-- P4: There are exactly four pibs
axiom P4 : ∃! (a b c d : Pib), ∀ p : Pib, p = a ∨ p = b ∨ p = c ∨ p = d

-- T1: There are exactly six maas
theorem T1 : ∃! (a b c d e f : Maa), ∀ m : Maa, m = a ∨ m = b ∨ m = c ∨ m = d ∨ m = e ∨ m = f :=
sorry

-- T2: There are exactly three maas in each pib
theorem T2 : ∀ p : Pib, ∃! (a b c : Maa), (∀ m : Maa, belongs_to m p ↔ (m = a ∨ m = b ∨ m = c)) :=
sorry

-- T3: For each maa there is exactly one other maa not in the same pib with it
theorem T3 : ∀ m1 : Maa, ∃! m2 : Maa, m1 ≠ m2 ∧ ∀ p : Pib, ¬(belongs_to m1 p ∧ belongs_to m2 p) :=
sorry

end T1_T2_T3_l1332_133201


namespace unique_valid_integer_l1332_133284

-- Define a type for 10-digit integers
def TenDigitInteger := Fin 10 → Fin 10

-- Define a property for strictly increasing sequence
def StrictlyIncreasing (n : TenDigitInteger) : Prop :=
  ∀ i j : Fin 10, i < j → n i < n j

-- Define a property for using each digit exactly once
def UsesEachDigitOnce (n : TenDigitInteger) : Prop :=
  ∀ d : Fin 10, ∃! i : Fin 10, n i = d

-- Define the set of valid integers
def ValidIntegers : Set TenDigitInteger :=
  {n | n 0 ≠ 0 ∧ StrictlyIncreasing n ∧ UsesEachDigitOnce n}

-- Theorem statement
theorem unique_valid_integer : ∃! n : TenDigitInteger, n ∈ ValidIntegers := by
  sorry

end unique_valid_integer_l1332_133284


namespace division_remainder_proof_l1332_133268

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 176) (h2 : divisor = 19) (h3 : quotient = 9) :
  dividend - divisor * quotient = 5 := by
sorry

end division_remainder_proof_l1332_133268


namespace square_areas_square_areas_concrete_l1332_133216

/-- Given a square with area 100, prove the areas of the inscribed square and right triangle --/
theorem square_areas (S : Real) (h1 : S^2 = 100) :
  let small_square_area := S^2 / 4
  let right_triangle_area := S^2 / 16
  (small_square_area = 50) ∧ (right_triangle_area = 12.5) := by
  sorry

/-- Alternative formulation using concrete numbers --/
theorem square_areas_concrete :
  let large_square_area := 100
  let small_square_area := large_square_area / 2
  let right_triangle_area := large_square_area / 8
  (small_square_area = 50) ∧ (right_triangle_area = 12.5) := by
  sorry

end square_areas_square_areas_concrete_l1332_133216


namespace longest_segment_in_cylinder_l1332_133264

/-- The longest segment in a cylinder -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 * Real.sqrt 2 := by
  sorry

end longest_segment_in_cylinder_l1332_133264


namespace smallest_sum_with_real_roots_l1332_133290

theorem smallest_sum_with_real_roots (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + a*x + 3*b = 0) → 
  (∃ x : ℝ, x^2 + 3*b*x + a = 0) → 
  a + b ≥ 7 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    (∃ x : ℝ, x^2 + a₀*x + 3*b₀ = 0) ∧ 
    (∃ x : ℝ, x^2 + 3*b₀*x + a₀ = 0) ∧ 
    a₀ + b₀ = 7 :=
by sorry

end smallest_sum_with_real_roots_l1332_133290


namespace min_tablets_extracted_l1332_133223

/-- Represents the number of tablets of each medicine type in the box -/
structure MedicineBox where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the minimum number of tablets to extract to guarantee at least three of each type -/
def minTablets (box : MedicineBox) : Nat :=
  (box.a + box.b + box.c) - min (box.a - 3) 0 - min (box.b - 3) 0 - min (box.c - 3) 0

/-- Theorem: The minimum number of tablets to extract from the given box is 48 -/
theorem min_tablets_extracted (box : MedicineBox) 
  (ha : box.a = 20) (hb : box.b = 25) (hc : box.c = 15) : 
  minTablets box = 48 := by
  sorry

end min_tablets_extracted_l1332_133223


namespace square_eq_sixteen_l1332_133214

theorem square_eq_sixteen (x : ℝ) : (x - 3)^2 = 16 ↔ x = 7 ∨ x = -1 := by
  sorry

end square_eq_sixteen_l1332_133214


namespace diagonal_lengths_and_t_value_l1332_133247

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-1, -2)
def C : ℝ × ℝ := (-2, -1)

-- Define vectors
def vec_AB : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)
def vec_OC : ℝ × ℝ := C

-- Calculate the fourth point D
def D : ℝ × ℝ := (A.1 + C.1 - B.1, A.2 + C.2 - B.2)

-- Define diagonals
def vec_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def vec_BD : ℝ × ℝ := (D.1 - B.1, D.2 - B.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statements
theorem diagonal_lengths_and_t_value : 
  (vec_AC.1^2 + vec_AC.2^2 = 32) ∧ 
  (vec_BD.1^2 + vec_BD.2^2 = 40) ∧ 
  (∃ t : ℝ, t = -11/5 ∧ dot_product (vec_AB.1 + t * vec_OC.1, vec_AB.2 + t * vec_OC.2) vec_OC = 0) :=
sorry

end diagonal_lengths_and_t_value_l1332_133247


namespace marbles_cost_l1332_133232

/-- The amount spent on marbles, given the total spent on toys and the cost of a football -/
def amount_spent_on_marbles (total_spent : ℝ) (football_cost : ℝ) : ℝ :=
  total_spent - football_cost

/-- Theorem stating that the amount spent on marbles is $6.59 -/
theorem marbles_cost (total_spent : ℝ) (football_cost : ℝ)
  (h1 : total_spent = 12.30)
  (h2 : football_cost = 5.71) :
  amount_spent_on_marbles total_spent football_cost = 6.59 := by
  sorry

end marbles_cost_l1332_133232


namespace height_difference_l1332_133252

theorem height_difference (height_A : ℝ) (initial_ratio : ℝ) (growth : ℝ) : 
  height_A = 72 →
  initial_ratio = 2/3 →
  growth = 10 →
  height_A - (initial_ratio * height_A + growth) = 14 := by
sorry

end height_difference_l1332_133252


namespace solution_set_theorem_l1332_133266

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the domain of f
def domain : Set ℝ := { x | x > 0 }

-- State the theorem
theorem solution_set_theorem 
  (h_deriv : ∀ x ∈ domain, HasDerivAt f (f' x) x)
  (h_ineq : ∀ x ∈ domain, f x < -x * f' x) :
  { x ∈ domain | f (x + 1) > (x - 1) * f (x^2 - 1) } = { x | x > 2 } := by
  sorry

end solution_set_theorem_l1332_133266


namespace sum_of_squares_l1332_133277

theorem sum_of_squares (x y : ℝ) (hx : x^2 = 8*x + y) (hy : y^2 = x + 8*y) (hxy : x ≠ y) :
  x^2 + y^2 = 63 := by
  sorry

end sum_of_squares_l1332_133277


namespace replacement_process_terminates_l1332_133254

/-- A finite sequence of zeros and ones -/
def BinarySequence := List Bool

/-- The operation of replacing "01" with "1000" in a binary sequence -/
def replace01With1000 (seq : BinarySequence) : BinarySequence :=
  match seq with
  | [] => []
  | [x] => [x]
  | false :: true :: xs => true :: false :: false :: false :: xs
  | x :: xs => x :: replace01With1000 xs

/-- The weight of a binary sequence -/
def weight (seq : BinarySequence) : Nat :=
  seq.foldl (λ acc x => if x then 4 * acc else acc + 1) 0

/-- Theorem: The replacement process will eventually terminate -/
theorem replacement_process_terminates (seq : BinarySequence) :
  ∃ n : Nat, ∀ m : Nat, m ≥ n → replace01With1000^[m] seq = replace01With1000^[n] seq :=
sorry

end replacement_process_terminates_l1332_133254


namespace square_diagonal_and_inscribed_circle_area_l1332_133207

/-- Given a square with side length 40√3 cm, this theorem proves the length of its diagonal
    and the area of its inscribed circle. -/
theorem square_diagonal_and_inscribed_circle_area 
  (side_length : ℝ) 
  (h_side : side_length = 40 * Real.sqrt 3) :
  ∃ (diagonal_length : ℝ) (inscribed_circle_area : ℝ),
    diagonal_length = 40 * Real.sqrt 6 ∧
    inscribed_circle_area = 1200 * Real.pi := by
  sorry


end square_diagonal_and_inscribed_circle_area_l1332_133207


namespace bill_difference_l1332_133218

theorem bill_difference (christine_tip : ℝ) (christine_percent : ℝ)
  (alex_tip : ℝ) (alex_percent : ℝ) :
  christine_tip = 3 →
  christine_percent = 15 →
  alex_tip = 4 →
  alex_percent = 10 →
  christine_tip = (christine_percent / 100) * christine_bill →
  alex_tip = (alex_percent / 100) * alex_bill →
  alex_bill - christine_bill = 20 :=
by
  sorry

end bill_difference_l1332_133218


namespace work_completion_time_l1332_133210

/-- Given a work that can be completed by person a in 15 days and by person b in 30 days,
    prove that a and b together can complete the work in 10 days. -/
theorem work_completion_time (work : ℝ) (a b : ℝ) 
    (ha : a * 15 = work) 
    (hb : b * 30 = work) : 
    (a + b) * 10 = work := by
  sorry

end work_completion_time_l1332_133210


namespace exponential_inequality_l1332_133230

theorem exponential_inequality (x y a b : ℝ) 
  (hxy : x > y ∧ y > 1) 
  (hab : 0 < a ∧ a < b ∧ b < 1) : 
  a^x < b^y := by
  sorry

end exponential_inequality_l1332_133230


namespace sqrt_equality_l1332_133212

theorem sqrt_equality (a b x : ℝ) (h1 : a < b) (h2 : -b ≤ x) (h3 : x ≤ -a) :
  Real.sqrt (-(x+a)^3*(x+b)) = -(x+a) * Real.sqrt (-(x+a)*(x+b)) :=
by sorry

end sqrt_equality_l1332_133212


namespace hawks_score_l1332_133281

/-- The number of touchdowns scored by the Hawks -/
def num_touchdowns : ℕ := 3

/-- The number of points per touchdown -/
def points_per_touchdown : ℕ := 7

/-- The total points scored by the Hawks -/
def total_points : ℕ := num_touchdowns * points_per_touchdown

theorem hawks_score :
  total_points = 21 :=
sorry

end hawks_score_l1332_133281


namespace pet_store_birds_l1332_133285

/-- The number of bird cages in the pet store -/
def num_cages : ℝ := 6.0

/-- The number of parrots in each cage -/
def parrots_per_cage : ℝ := 6.0

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℝ := 2.0

/-- The total number of birds in the pet store -/
def total_birds : ℝ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds : total_birds = 48.0 := by
  sorry

end pet_store_birds_l1332_133285


namespace tom_games_owned_before_l1332_133261

/-- The number of games Tom owned before purchasing new games -/
def games_owned_before : ℕ := 0

/-- The cost of the Batman game in dollars -/
def batman_game_cost : ℚ := 13.60

/-- The cost of the Superman game in dollars -/
def superman_game_cost : ℚ := 5.06

/-- The total amount Tom spent on video games in dollars -/
def total_spent : ℚ := 18.66

theorem tom_games_owned_before :
  games_owned_before = 0 ∧
  batman_game_cost + superman_game_cost = total_spent :=
sorry

end tom_games_owned_before_l1332_133261


namespace matrix_commutation_fraction_l1332_133299

/-- Given two matrices A and B, where A is fixed and B has variable entries,
    prove that if A * B = B * A and 3b ≠ c, then (a - d) / (c - 3b) = 1. -/
theorem matrix_commutation_fraction (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 4]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (3 * b ≠ c) → ((a - d) / (c - 3 * b) = 1) := by
  sorry

end matrix_commutation_fraction_l1332_133299


namespace total_profit_is_100_l1332_133253

/-- Calculates the total profit given investments and A's profit share -/
def calculate_total_profit (a_investment : ℕ) (a_months : ℕ) (b_investment : ℕ) (b_months : ℕ) (a_profit_share : ℕ) : ℕ :=
  let a_investment_ratio := a_investment * a_months
  let b_investment_ratio := b_investment * b_months
  let total_investment_ratio := a_investment_ratio + b_investment_ratio
  (a_profit_share * total_investment_ratio) / a_investment_ratio

/-- Proves that the total profit is $100 given the specified investments and A's profit share -/
theorem total_profit_is_100 :
  calculate_total_profit 100 12 200 6 50 = 100 := by
  sorry

end total_profit_is_100_l1332_133253


namespace solve_linear_equation_l1332_133298

theorem solve_linear_equation (x : ℝ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 := by
  sorry

end solve_linear_equation_l1332_133298


namespace band_sections_fraction_l1332_133263

theorem band_sections_fraction (trumpet_fraction trombone_fraction : ℝ) 
  (h1 : trumpet_fraction = 0.5)
  (h2 : trombone_fraction = 0.125) :
  trumpet_fraction + trombone_fraction = 0.625 := by
  sorry

end band_sections_fraction_l1332_133263


namespace simplify_and_evaluate_l1332_133237

theorem simplify_and_evaluate (x : ℤ) (h1 : -2 < x) (h2 : x < 3) :
  (x / (x + 1) - 3 * x / (x - 1)) / (x / (x^2 - 1)) = -8 := by
  sorry

end simplify_and_evaluate_l1332_133237


namespace square_area_ratio_l1332_133274

theorem square_area_ratio (x : ℝ) (h : x > 0) : 
  (x^2) / ((4*x)^2) = 1 / 16 := by
  sorry

end square_area_ratio_l1332_133274


namespace customers_who_tried_sample_l1332_133231

/-- Given a store that puts out product samples, this theorem calculates
    the number of customers who tried a sample based on the given conditions. -/
theorem customers_who_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left : ℕ)
  (h1 : samples_per_box = 20)
  (h2 : boxes_opened = 12)
  (h3 : samples_left = 5) :
  samples_per_box * boxes_opened - samples_left = 235 :=
by sorry

end customers_who_tried_sample_l1332_133231


namespace bread_slices_left_l1332_133294

theorem bread_slices_left (
  initial_slices : Nat) 
  (days_in_week : Nat)
  (slices_per_sandwich : Nat)
  (extra_sandwiches : Nat) :
  initial_slices = 22 →
  days_in_week = 7 →
  slices_per_sandwich = 2 →
  extra_sandwiches = 1 →
  initial_slices - (days_in_week + extra_sandwiches) * slices_per_sandwich = 6 :=
by sorry

end bread_slices_left_l1332_133294


namespace max_sum_of_squares_max_sum_of_squares_value_exact_max_sum_of_squares_l1332_133250

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 95 →
  a * d + b * c = 180 →
  c * d = 105 →
  ∀ (w x y z : ℝ), 
    w + x = 18 →
    w * x + y + z = 95 →
    w * z + x * y = 180 →
    y * z = 105 →
    a^2 + b^2 + c^2 + d^2 ≥ w^2 + x^2 + y^2 + z^2 :=
by
  sorry

theorem max_sum_of_squares_value (a b c d : ℝ) :
  a + b = 18 →
  a * b + c + d = 95 →
  a * d + b * c = 180 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 1486 :=
by
  sorry

theorem exact_max_sum_of_squares (a b c d : ℝ) :
  a + b = 18 →
  a * b + c + d = 95 →
  a * d + b * c = 180 →
  c * d = 105 →
  ∃ (w x y z : ℝ),
    w + x = 18 ∧
    w * x + y + z = 95 ∧
    w * z + x * y = 180 ∧
    y * z = 105 ∧
    w^2 + x^2 + y^2 + z^2 = 1486 :=
by
  sorry

end max_sum_of_squares_max_sum_of_squares_value_exact_max_sum_of_squares_l1332_133250


namespace red_balls_count_l1332_133260

theorem red_balls_count (yellow_balls : ℕ) (total_balls : ℕ) 
  (yellow_prob : ℚ) (h1 : yellow_balls = 4) 
  (h2 : yellow_prob = 1 / 5) 
  (h3 : yellow_prob = yellow_balls / total_balls) : 
  total_balls - yellow_balls = 16 := by
  sorry

end red_balls_count_l1332_133260


namespace blue_eyes_count_l1332_133200

/-- The number of people in the theater -/
def total_people : ℕ := 100

/-- The number of people with brown eyes -/
def brown_eyes : ℕ := total_people / 2

/-- The number of people with black eyes -/
def black_eyes : ℕ := total_people / 4

/-- The number of people with green eyes -/
def green_eyes : ℕ := 6

/-- The number of people with blue eyes -/
def blue_eyes : ℕ := total_people - (brown_eyes + black_eyes + green_eyes)

theorem blue_eyes_count : blue_eyes = 19 := by
  sorry

end blue_eyes_count_l1332_133200


namespace cube_difference_l1332_133289

theorem cube_difference (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 35) : a^3 - b^3 = 200 := by
  sorry

end cube_difference_l1332_133289


namespace remaining_boys_average_weight_l1332_133219

/-- The average weight of the remaining 8 boys given the following conditions:
  - There are 20 boys with an average weight of 50.25 kg
  - There are 8 remaining boys
  - The average weight of all 28 boys is 48.792857142857144 kg
-/
theorem remaining_boys_average_weight :
  let num_group1 : ℕ := 20
  let avg_group1 : ℝ := 50.25
  let num_group2 : ℕ := 8
  let total_num : ℕ := num_group1 + num_group2
  let total_avg : ℝ := 48.792857142857144
  
  ((num_group1 : ℝ) * avg_group1 + (num_group2 : ℝ) * avg_group2) / (total_num : ℝ) = total_avg →
  avg_group2 = 45.15
  := by sorry

end remaining_boys_average_weight_l1332_133219


namespace plot_length_is_56_l1332_133225

/-- Proves that the length of a rectangular plot is 56 meters given the specified conditions -/
theorem plot_length_is_56 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 12 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.5 →
  total_cost = 5300 →
  total_cost = perimeter * cost_per_meter →
  length = 56 := by
sorry

end plot_length_is_56_l1332_133225


namespace cow_plus_cow_equals_milk_l1332_133222

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents an assignment of digits to letters -/
structure LetterAssignment where
  C : Digit
  O : Digit
  W : Digit
  M : Digit
  I : Digit
  L : Digit
  K : Digit
  all_different : C ≠ O ∧ C ≠ W ∧ C ≠ M ∧ C ≠ I ∧ C ≠ L ∧ C ≠ K ∧
                  O ≠ W ∧ O ≠ M ∧ O ≠ I ∧ O ≠ L ∧ O ≠ K ∧
                  W ≠ M ∧ W ≠ I ∧ W ≠ L ∧ W ≠ K ∧
                  M ≠ I ∧ M ≠ L ∧ M ≠ K ∧
                  I ≠ L ∧ I ≠ K ∧
                  L ≠ K

/-- Converts a LetterAssignment to the numeric value of COW -/
def cow_value (assignment : LetterAssignment) : ℕ :=
  100 * assignment.C.val + 10 * assignment.O.val + assignment.W.val

/-- Converts a LetterAssignment to the numeric value of MILK -/
def milk_value (assignment : LetterAssignment) : ℕ :=
  1000 * assignment.M.val + 100 * assignment.I.val + 10 * assignment.L.val + assignment.K.val

/-- The main theorem stating that there are exactly three solutions to the puzzle -/
theorem cow_plus_cow_equals_milk :
  ∃! (solutions : Finset LetterAssignment),
    solutions.card = 3 ∧
    (∀ assignment ∈ solutions, 2 * cow_value assignment = milk_value assignment) :=
sorry

end cow_plus_cow_equals_milk_l1332_133222


namespace negative_solutions_count_l1332_133286

def f (x : ℤ) : ℤ := x^6 - 75*x^4 + 1000*x^2 - 6000

theorem negative_solutions_count :
  ∃! (S : Finset ℤ), (∀ x ∈ S, f x < 0) ∧ (∀ x ∉ S, f x ≥ 0) ∧ Finset.card S = 12 := by
  sorry

end negative_solutions_count_l1332_133286


namespace president_vice_selection_ways_l1332_133238

/-- The number of ways to choose a president and vice-president from a club with the given conditions -/
def choose_president_and_vice (total_members boys girls : ℕ) : ℕ :=
  (boys * (boys - 1)) + (girls * (girls - 1))

/-- Theorem stating the number of ways to choose a president and vice-president under the given conditions -/
theorem president_vice_selection_ways :
  let total_members : ℕ := 30
  let boys : ℕ := 18
  let girls : ℕ := 12
  choose_president_and_vice total_members boys girls = 438 := by
  sorry

#eval choose_president_and_vice 30 18 12

end president_vice_selection_ways_l1332_133238


namespace imaginary_unit_equation_l1332_133259

/-- Given that i is the imaginary unit and |((a+i)/i)| = 2, prove that a = √3 where a is a positive real number. -/
theorem imaginary_unit_equation (i : ℂ) (a : ℝ) (h1 : i * i = -1) (h2 : a > 0) :
  Complex.abs ((a + i) / i) = 2 → a = Real.sqrt 3 := by
  sorry

end imaginary_unit_equation_l1332_133259


namespace geometric_arithmetic_sequence_sum_l1332_133296

theorem geometric_arithmetic_sequence_sum (x y : ℝ) : 
  5 < x ∧ x < y ∧ y < 15 →
  (∃ r : ℝ, r > 0 ∧ x = 5 * r ∧ y = 5 * r^2) →
  (∃ d : ℝ, y = x + d ∧ 15 = y + d) →
  x + y = 10 := by
sorry

end geometric_arithmetic_sequence_sum_l1332_133296


namespace weeks_of_papayas_l1332_133258

def jake_papayas_per_week : ℕ := 3
def brother_papayas_per_week : ℕ := 5
def father_papayas_per_week : ℕ := 4
def total_papayas_bought : ℕ := 48

theorem weeks_of_papayas : 
  (total_papayas_bought / (jake_papayas_per_week + brother_papayas_per_week + father_papayas_per_week) = 4) := by
  sorry

end weeks_of_papayas_l1332_133258


namespace exists_fib_divisible_l1332_133276

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- For any natural number m, there exists a Fibonacci number divisible by m -/
theorem exists_fib_divisible (m : ℕ) : ∃ n : ℕ, n ≥ 1 ∧ m ∣ fib n := by
  sorry

end exists_fib_divisible_l1332_133276


namespace litter_patrol_collection_l1332_133205

theorem litter_patrol_collection (glass_bottles : ℕ) (aluminum_cans : ℕ) : 
  glass_bottles = 10 → aluminum_cans = 8 → glass_bottles + aluminum_cans = 18 := by
  sorry

end litter_patrol_collection_l1332_133205


namespace reciprocal_of_neg_tan_60_l1332_133270

theorem reciprocal_of_neg_tan_60 :
  (-(Real.tan (60 * π / 180)))⁻¹ = -((3 : ℝ).sqrt / 3) := by
  sorry

end reciprocal_of_neg_tan_60_l1332_133270


namespace cos_x_plus_2y_eq_one_l1332_133278

/-- Given two real numbers x and y satisfying specific equations, prove that cos(x + 2y) = 1 -/
theorem cos_x_plus_2y_eq_one (x y : ℝ) 
  (hx : x^3 + Real.cos x + x - 2 = 0)
  (hy : 8 * y^3 - 2 * (Real.cos y)^2 + 2 * y + 3 = 0) :
  Real.cos (x + 2 * y) = 1 := by
  sorry

end cos_x_plus_2y_eq_one_l1332_133278


namespace inequality_proof_l1332_133292

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (1 + 2*a)) + (1 / (1 + 2*b)) + (1 / (1 + 2*c)) ≥ 1 := by
  sorry

end inequality_proof_l1332_133292


namespace circle_satisfies_conditions_l1332_133224

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define our circle
def our_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define what it means for two circles to be tangent
def tangent (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, f x y ∧ g x y ∧ 
  ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
    ((x' - x)^2 + (y' - y)^2 < δ^2) → 
    (f x' y' ↔ g x' y') → (x' = x ∧ y' = y)

theorem circle_satisfies_conditions : 
  our_circle 3 1 ∧ 
  our_circle 1 1 ∧ 
  tangent our_circle C1 := by sorry

end circle_satisfies_conditions_l1332_133224


namespace kddk_divisible_by_7_l1332_133206

/-- Represents a base-6 digit -/
def Base6Digit : Type := { n : ℕ // n < 6 }

/-- Converts a base-6 number of the form kddk to base 10 -/
def toBase10 (k d : Base6Digit) : ℕ :=
  217 * k.val + 42 * d.val

theorem kddk_divisible_by_7 (k d : Base6Digit) :
  7 ∣ toBase10 k d ↔ k = d :=
sorry

end kddk_divisible_by_7_l1332_133206


namespace quadratic_shift_l1332_133211

/-- Represents a quadratic function of the form y = -(x+a)^2 + b -/
def QuadraticFunction (a b : ℝ) := λ x : ℝ => -(x + a)^2 + b

/-- Represents a horizontal shift of a function -/
def HorizontalShift (f : ℝ → ℝ) (shift : ℝ) := λ x : ℝ => f (x - shift)

/-- Theorem: Shifting the graph of y = -(x+2)^2 + 1 by 1 unit to the right 
    results in the function y = -(x+1)^2 + 1 -/
theorem quadratic_shift :
  HorizontalShift (QuadraticFunction 2 1) 1 = QuadraticFunction 1 1 := by
  sorry

end quadratic_shift_l1332_133211


namespace solution_to_equation_l1332_133245

theorem solution_to_equation : ∃ x : ℝ, 4*x + 9*x = 360 - 9*(x - 4) ∧ x = 18 := by
  sorry

end solution_to_equation_l1332_133245


namespace C_power_50_is_identity_l1332_133242

def C : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 2],
    ![-8, -5]]

theorem C_power_50_is_identity :
  C ^ 50 = (1 : Matrix (Fin 2) (Fin 2) ℤ) := by
  sorry

end C_power_50_is_identity_l1332_133242


namespace mitch_boat_financing_l1332_133220

/-- The amount Mitch has saved in dollars -/
def total_savings : ℕ := 20000

/-- The cost of a new boat per foot in dollars -/
def boat_cost_per_foot : ℕ := 1500

/-- The maximum length of boat Mitch can buy in feet -/
def max_boat_length : ℕ := 12

/-- The amount Mitch needs to keep for license and registration in dollars -/
def license_registration_cost : ℕ := 500

/-- The ratio of docking fees to license and registration cost -/
def docking_fee_ratio : ℕ := 3

theorem mitch_boat_financing :
  license_registration_cost * (docking_fee_ratio + 1) = 
    total_savings - (boat_cost_per_foot * max_boat_length) :=
by sorry

end mitch_boat_financing_l1332_133220


namespace negation_of_universal_proposition_l1332_133297

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end negation_of_universal_proposition_l1332_133297


namespace tangent_line_equation_l1332_133280

-- Define the function f(x) = x^3 + x
def f (x : ℝ) := x^3 + x

-- Define the derivative of f(x)
def f' (x : ℝ) := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (4 * x - y - 2 = 0) :=
by sorry

end tangent_line_equation_l1332_133280


namespace fathers_children_l1332_133267

theorem fathers_children (father_age : ℕ) (children_sum : ℕ) (n : ℕ) : 
  father_age = 75 →
  father_age = children_sum →
  children_sum + 15 * n = 2 * (father_age + 15) →
  n = 7 := by
sorry

end fathers_children_l1332_133267


namespace quadratic_inequality_equivalence_l1332_133283

theorem quadratic_inequality_equivalence (x : ℝ) :
  x^2 + 5*x - 14 < 0 ↔ -7 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_equivalence_l1332_133283


namespace complex_equation_solution_l1332_133221

theorem complex_equation_solution (a : ℝ) : 
  Complex.abs ((a + Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 ∨ a = -Real.sqrt 3 := by
  sorry

end complex_equation_solution_l1332_133221


namespace percentage_of_indian_men_l1332_133233

theorem percentage_of_indian_men (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (percentage_indian_women : ℚ) (percentage_indian_children : ℚ)
  (percentage_not_indian : ℚ) :
  total_men = 500 →
  total_women = 300 →
  total_children = 500 →
  percentage_indian_women = 60 / 100 →
  percentage_indian_children = 70 / 100 →
  percentage_not_indian = 55.38461538461539 / 100 →
  (total_men * (10 / 100) + total_women * percentage_indian_women + total_children * percentage_indian_children : ℚ) =
  (total_men + total_women + total_children : ℕ) * (1 - percentage_not_indian) :=
by sorry

end percentage_of_indian_men_l1332_133233


namespace max_value_g_geq_seven_l1332_133269

theorem max_value_g_geq_seven (a b : ℝ) (h_a : a ≤ -1) : 
  let f := fun x : ℝ => Real.exp x * (x^2 + a*x + 1)
  let g := fun x : ℝ => 2*x^3 + 3*(b+1)*x^2 + 6*b*x + 6
  let x_min_f := -(a + 1)
  (∀ x : ℝ, g x ≥ g x_min_f) → 
  (∀ x : ℝ, f x ≥ f x_min_f) → 
  ∃ x : ℝ, g x ≥ 7 :=
by sorry

end max_value_g_geq_seven_l1332_133269


namespace dice_sum_probability_l1332_133209

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The target sum we're aiming for -/
def target_sum : ℕ := 15

/-- 
The number of ways to achieve the target sum when rolling the specified number of dice.
This is equivalent to the coefficient of x^target_sum in the expansion of (x + x^2 + ... + x^num_faces)^num_dice.
-/
def num_ways_to_achieve_sum : ℕ := 2002

theorem dice_sum_probability : 
  num_ways_to_achieve_sum = 2002 := by sorry

end dice_sum_probability_l1332_133209


namespace ac_circuit_current_l1332_133271

def V : ℂ := 2 + 2*Complex.I
def Z : ℂ := 2 - 2*Complex.I

theorem ac_circuit_current : V = Complex.I * Z := by sorry

end ac_circuit_current_l1332_133271


namespace one_thirds_in_nine_halves_l1332_133236

theorem one_thirds_in_nine_halves : (9 : ℚ) / 2 / (1 / 3) = 27 / 2 := by
  sorry

end one_thirds_in_nine_halves_l1332_133236


namespace self_centered_max_solutions_l1332_133213

/-- A polynomial is self-centered if it has integer coefficients and p(200) = 200 -/
def SelfCentered (p : ℤ → ℤ) : Prop :=
  (∀ x, ∃ n : ℕ, p x = (x : ℤ) ^ n) ∧ p 200 = 200

/-- The main theorem: any self-centered polynomial has at most 10 integer solutions to p(k) = k^4 -/
theorem self_centered_max_solutions (p : ℤ → ℤ) (h : SelfCentered p) :
  ∃ s : Finset ℤ, s.card ≤ 10 ∧ ∀ k : ℤ, p k = k^4 → k ∈ s := by
  sorry

end self_centered_max_solutions_l1332_133213


namespace line_plane_perpendicularity_l1332_133203

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → perp α β :=
sorry

end line_plane_perpendicularity_l1332_133203


namespace original_speed_is_30_l1332_133295

/-- Represents the driving scenario with given conditions -/
def DrivingScenario (original_speed : ℝ) : Prop :=
  let total_distance : ℝ := 100
  let breakdown_time : ℝ := 2
  let repair_time : ℝ := 0.5
  let speed_increase_factor : ℝ := 1.6
  
  -- Time equation: total time = time before breakdown + repair time + time after repair
  total_distance / original_speed = 
    breakdown_time + repair_time + 
    (total_distance - breakdown_time * original_speed) / (speed_increase_factor * original_speed)

/-- Theorem stating that the original speed satisfying the driving scenario is 30 km/h -/
theorem original_speed_is_30 : 
  ∃ (speed : ℝ), DrivingScenario speed ∧ speed = 30 := by
  sorry

end original_speed_is_30_l1332_133295


namespace no_valid_arrangement_l1332_133246

def is_valid_arrangement (arr : List Nat) : Prop :=
  ∀ i, i < arr.length - 1 → (10 * arr[i]! + arr[i+1]!) % 7 = 0

theorem no_valid_arrangement : 
  ¬ ∃ (arr : List Nat), arr.toFinset = {1, 2, 3, 4, 5, 6, 8, 9} ∧ is_valid_arrangement arr :=
by sorry

end no_valid_arrangement_l1332_133246


namespace min_triangle_area_on_unit_grid_l1332_133249

/-- The area of a triangle given three points on a 2D grid -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℤ) : ℚ :=
  (1 / 2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- The minimum area of a triangle on a unit grid -/
theorem min_triangle_area_on_unit_grid : 
  ∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    triangleArea x1 y1 x2 y2 x3 y3 = (1 / 2 : ℚ) ∧ 
    (∀ (a1 b1 a2 b2 a3 b3 : ℤ), triangleArea a1 b1 a2 b2 a3 b3 ≥ (1 / 2 : ℚ)) :=
sorry

end min_triangle_area_on_unit_grid_l1332_133249


namespace ivans_chess_claim_impossible_l1332_133244

theorem ivans_chess_claim_impossible : ¬ ∃ (n : ℕ), n > 0 ∧ n + 3*n + 6*n = 64 := by sorry

end ivans_chess_claim_impossible_l1332_133244


namespace line_perp_to_plane_perp_to_line_in_plane_l1332_133291

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- State the theorem
theorem line_perp_to_plane_perp_to_line_in_plane
  (a b : Line) (α : Plane)
  (h1 : perp_line_plane a α)
  (h2 : subset_line_plane b α) :
  perp_line_line a b :=
sorry

end line_perp_to_plane_perp_to_line_in_plane_l1332_133291


namespace absent_men_count_l1332_133243

/-- Proves the number of absent men in a work group --/
theorem absent_men_count (total_men : ℕ) (original_days : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 20)
  (h2 : original_days = 20)
  (h3 : actual_days = 40)
  (h4 : total_men * original_days = (total_men - absent_men) * actual_days) :
  absent_men = 10 := by
  sorry

#check absent_men_count

end absent_men_count_l1332_133243


namespace candy_distribution_l1332_133257

theorem candy_distribution (num_clowns num_children initial_candies remaining_candies : ℕ) 
  (h1 : num_clowns = 4)
  (h2 : num_children = 30)
  (h3 : initial_candies = 700)
  (h4 : remaining_candies = 20)
  (h5 : ∃ (candies_per_person : ℕ), 
    (num_clowns + num_children) * candies_per_person = initial_candies - remaining_candies) :
  ∃ (candies_per_person : ℕ), candies_per_person = 20 ∧ 
    (num_clowns + num_children) * candies_per_person = initial_candies - remaining_candies :=
by
  sorry

end candy_distribution_l1332_133257


namespace complement_A_inter_B_l1332_133228

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x * (x - 3) ≥ 0}
def B : Set ℝ := {x | x ≤ 2}

-- State the theorem
theorem complement_A_inter_B :
  (Set.compl A) ∩ B = Set.Ioo 0 2 := by sorry

end complement_A_inter_B_l1332_133228


namespace plate_on_square_table_l1332_133273

/-- The distance from the edge of a round plate to the bottom edge of a square table -/
def plate_to_bottom_edge (top_margin left_margin right_margin : ℝ) : ℝ :=
  left_margin + right_margin - top_margin

theorem plate_on_square_table 
  (top_margin left_margin right_margin : ℝ) 
  (h_top : top_margin = 10)
  (h_left : left_margin = 63)
  (h_right : right_margin = 20) :
  plate_to_bottom_edge top_margin left_margin right_margin = 73 := by
sorry

end plate_on_square_table_l1332_133273


namespace reservoir_capacity_shortage_l1332_133240

/-- Proves that the normal level of a reservoir is 7 million gallons short of total capacity
    given specific conditions about the current amount and capacity. -/
theorem reservoir_capacity_shortage :
  ∀ (current_amount normal_level total_capacity : ℝ),
  current_amount = 6 →
  current_amount = 2 * normal_level →
  current_amount = 0.6 * total_capacity →
  total_capacity - normal_level = 7 := by
sorry

end reservoir_capacity_shortage_l1332_133240


namespace customer_ratio_l1332_133248

/-- The number of customers during breakfast on Friday -/
def breakfast_customers : ℕ := 73

/-- The number of customers during lunch on Friday -/
def lunch_customers : ℕ := 127

/-- The number of customers during dinner on Friday -/
def dinner_customers : ℕ := 87

/-- The predicted number of customers for Saturday -/
def predicted_saturday_customers : ℕ := 574

/-- The total number of customers on Friday -/
def friday_customers : ℕ := breakfast_customers + lunch_customers + dinner_customers

/-- The theorem stating the ratio of predicted Saturday customers to Friday customers -/
theorem customer_ratio : 
  (predicted_saturday_customers : ℚ) / (friday_customers : ℚ) = 574 / 287 := by
  sorry


end customer_ratio_l1332_133248


namespace third_term_of_geometric_sequence_l1332_133279

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem third_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 8)
  (h_a5 : a 5 = 64) :
  a 3 = 16 := by
sorry

end third_term_of_geometric_sequence_l1332_133279


namespace square_area_from_adjacent_vertices_l1332_133287

/-- The area of a square with adjacent vertices at (1, -2) and (-3, 5) is 65 -/
theorem square_area_from_adjacent_vertices : 
  let p1 : ℝ × ℝ := (1, -2)
  let p2 : ℝ × ℝ := (-3, 5)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  side_length^2 = 65 := by
  sorry

end square_area_from_adjacent_vertices_l1332_133287


namespace quasi_pythagorean_prime_divisor_l1332_133227

theorem quasi_pythagorean_prime_divisor (a b c : ℕ+) :
  c^2 = a^2 + b^2 + a*b → ∃ p : ℕ, p.Prime ∧ p > 5 ∧ p ∣ c := by
  sorry

end quasi_pythagorean_prime_divisor_l1332_133227


namespace min_value_fourth_root_plus_reciprocal_l1332_133241

theorem min_value_fourth_root_plus_reciprocal (x : ℝ) (hx : x > 0) :
  2 * x^(1/4) + 1/x ≥ 3 ∧ (2 * x^(1/4) + 1/x = 3 ↔ x = 1) :=
by sorry

end min_value_fourth_root_plus_reciprocal_l1332_133241


namespace gifted_books_count_l1332_133255

def books_per_month : ℕ := 2
def months_per_year : ℕ := 12
def bought_books : ℕ := 8
def reread_old_books : ℕ := 4

def borrowed_books : ℕ := bought_books - 2

def total_books_needed : ℕ := books_per_month * months_per_year
def new_books_needed : ℕ := total_books_needed - reread_old_books
def new_books_acquired : ℕ := bought_books + borrowed_books

theorem gifted_books_count : new_books_needed - new_books_acquired = 6 := by
  sorry

end gifted_books_count_l1332_133255


namespace no_rational_square_in_sequence_l1332_133275

def sequence_a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => sequence_a n + 2 / sequence_a n

theorem no_rational_square_in_sequence :
  ∀ n : ℕ, ¬ ∃ r : ℚ, sequence_a n = r ^ 2 := by
  sorry

end no_rational_square_in_sequence_l1332_133275


namespace successive_discounts_equivalence_l1332_133293

/-- Proves that three successive discounts are equivalent to a single discount -/
theorem successive_discounts_equivalence (original_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) (equivalent_discount : ℝ) : 
  original_price = 60 ∧ 
  discount1 = 0.15 ∧ 
  discount2 = 0.10 ∧ 
  discount3 = 0.20 ∧ 
  equivalent_discount = 0.388 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 
  original_price * (1 - equivalent_discount) :=
by sorry

end successive_discounts_equivalence_l1332_133293
