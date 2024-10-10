import Mathlib

namespace line_intersection_plane_parallel_l3487_348742

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the intersection of lines
variable (intersect : Line → Line → Prop)

-- Define parallel planes
variable (parallel : Plane → Plane → Prop)

-- Define the statement
theorem line_intersection_plane_parallel 
  (l m : Line) (α β : Plane) 
  (h1 : subset l α) (h2 : subset m β) :
  (¬ intersect l m → parallel α β) ∧ 
  ¬ (¬ intersect l m → parallel α β) ∧ 
  (parallel α β → ¬ intersect l m) :=
sorry

end line_intersection_plane_parallel_l3487_348742


namespace tangent_line_equation_l3487_348703

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The equation of a line -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ
  h_not_zero : A ≠ 0 ∨ B ≠ 0

/-- Theorem: The equation of the tangent line at a point on an ellipse -/
theorem tangent_line_equation (e : Ellipse) (p : PointOnEllipse e) :
  ∃ (l : Line), l.A = p.x / e.a^2 ∧ l.B = p.y / e.b^2 ∧ l.C = -1 ∧
  (∀ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 → l.A * x + l.B * y + l.C = 0 → x = p.x ∧ y = p.y) :=
sorry

end tangent_line_equation_l3487_348703


namespace sum_of_w_and_y_l3487_348749

theorem sum_of_w_and_y (W X Y Z : ℤ) : 
  W ∈ ({1, 2, 5, 6} : Set ℤ) →
  X ∈ ({1, 2, 5, 6} : Set ℤ) →
  Y ∈ ({1, 2, 5, 6} : Set ℤ) →
  Z ∈ ({1, 2, 5, 6} : Set ℤ) →
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
  (W : ℚ) / X + (Y : ℚ) / Z = 3 →
  W + Y = 8 := by
sorry

end sum_of_w_and_y_l3487_348749


namespace polynomial_characterization_l3487_348721

theorem polynomial_characterization (p : ℕ) (hp : Nat.Prime p) :
  ∀ (P : ℕ → ℕ),
    (∀ x : ℕ, x > 0 → P x > x) →
    (∀ m : ℕ, m > 0 → ∃ l : ℕ, m ∣ (Nat.iterate P l p)) →
    (∃ a b : ℕ → ℕ, (∀ x, P x = x + 1 ∨ P x = x + p) ∧
                    (∀ x, a x = x + 1) ∧
                    (∀ x, b x = x + p)) :=
by sorry

end polynomial_characterization_l3487_348721


namespace lcm_gcd_equality_l3487_348725

/-- For positive integers a, b, c, prove that 
    [a,b,c]^2 / ([a,b][b,c][c,a]) = (a,b,c)^2 / ((a,b)(b,c)(c,a)) -/
theorem lcm_gcd_equality (a b c : ℕ+) : 
  (Nat.lcm (Nat.lcm a b) c)^2 / (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a) = 
  (Nat.gcd (Nat.gcd a b) c)^2 / (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a) := by
  sorry

end lcm_gcd_equality_l3487_348725


namespace total_vehicles_l3487_348771

theorem total_vehicles (motorcycles bicycles : ℕ) 
  (h1 : motorcycles = 2) 
  (h2 : bicycles = 5) : 
  motorcycles + bicycles = 7 :=
by sorry

end total_vehicles_l3487_348771


namespace sum_of_reciprocals_l3487_348740

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  1 / x + 1 / y = 2 := by
  sorry

end sum_of_reciprocals_l3487_348740


namespace three_million_twenty_one_thousand_scientific_notation_l3487_348762

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem three_million_twenty_one_thousand_scientific_notation :
  toScientificNotation 3021000 = ScientificNotation.mk 3.021 6 (by sorry) :=
sorry

end three_million_twenty_one_thousand_scientific_notation_l3487_348762


namespace vector_triangle_inequality_l3487_348738

/-- Given two vectors AB and AC in a Euclidean space, with |AB| = 3 and |AC| = 6,
    prove that the magnitude of BC is between 3 and 9 inclusive. -/
theorem vector_triangle_inequality (A B C : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ‖B - A‖ = 3) (h2 : ‖C - A‖ = 6) : 
  3 ≤ ‖C - B‖ ∧ ‖C - B‖ ≤ 9 := by
  sorry

end vector_triangle_inequality_l3487_348738


namespace wood_sawed_off_l3487_348793

theorem wood_sawed_off (original_length final_length : ℝ) 
  (h1 : original_length = 0.41)
  (h2 : final_length = 0.08) :
  original_length - final_length = 0.33 := by
  sorry

end wood_sawed_off_l3487_348793


namespace bert_shopping_trip_l3487_348794

theorem bert_shopping_trip (initial_amount : ℝ) : 
  initial_amount = 52 →
  let hardware_spend := initial_amount / 4
  let after_hardware := initial_amount - hardware_spend
  let dryclean_spend := 9
  let after_dryclean := after_hardware - dryclean_spend
  let grocery_spend := after_dryclean / 2
  let final_amount := after_dryclean - grocery_spend
  final_amount = 15 := by
sorry

end bert_shopping_trip_l3487_348794


namespace prob_heart_spade_king_two_draws_l3487_348732

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ := 52)
  (target_cards : ℕ := 28)

/-- Calculates the probability of drawing at least one target card in two draws with replacement -/
def prob_at_least_one_target (d : Deck) : ℚ :=
  1 - (1 - d.target_cards / d.total_cards) ^ 2

/-- The probability of drawing at least one heart, spade, or king in two draws with replacement from a standard deck is 133/169 -/
theorem prob_heart_spade_king_two_draws :
  prob_at_least_one_target (Deck.mk 52 28) = 133 / 169 := by
  sorry

#eval prob_at_least_one_target (Deck.mk 52 28)

end prob_heart_spade_king_two_draws_l3487_348732


namespace parabola_x_intercepts_l3487_348713

/-- The number of x-intercepts of the parabola x = -3y^2 + 2y + 2 -/
theorem parabola_x_intercepts :
  let f : ℝ → ℝ := λ y => -3 * y^2 + 2 * y + 2
  ∃! x : ℝ, ∃ y : ℝ, f y = x ∧ y = 0 :=
by sorry

end parabola_x_intercepts_l3487_348713


namespace sufficient_not_necessary_l3487_348765

theorem sufficient_not_necessary (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x - y ≥ 2 → 2^x - 2^y ≥ 3) ∧
  (∃ x y, 2^x - 2^y ≥ 3 ∧ x - y < 2) :=
by sorry

end sufficient_not_necessary_l3487_348765


namespace farm_hens_count_l3487_348797

/-- Represents the number of animals on a farm. -/
structure FarmAnimals where
  hens : ℕ
  cows : ℕ
  goats : ℕ

/-- Calculates the total number of heads for all animals on the farm. -/
def totalHeads (farm : FarmAnimals) : ℕ :=
  farm.hens + farm.cows + farm.goats

/-- Calculates the total number of feet for all animals on the farm. -/
def totalFeet (farm : FarmAnimals) : ℕ :=
  2 * farm.hens + 4 * farm.cows + 4 * farm.goats

/-- Theorem stating that given the conditions, there are 66 hens on the farm. -/
theorem farm_hens_count (farm : FarmAnimals) 
  (head_count : totalHeads farm = 120) 
  (feet_count : totalFeet farm = 348) : 
  farm.hens = 66 := by
  sorry

end farm_hens_count_l3487_348797


namespace geometric_arithmetic_relation_l3487_348739

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_relation (a b : ℕ → ℝ) :
  geometric_sequence a
  → a 2 = 4
  → a 4 = 16
  → arithmetic_sequence b
  → b 3 = a 3
  → b 5 = a 5
  → ∀ n : ℕ, b n = 12 * n - 28 :=
by
  sorry

end geometric_arithmetic_relation_l3487_348739


namespace base_number_proof_l3487_348702

theorem base_number_proof (k : ℕ) (x : ℤ) 
  (h1 : 21^k ∣ 435961)
  (h2 : x^k - k^7 = 1) :
  x = 2 :=
sorry

end base_number_proof_l3487_348702


namespace total_covered_area_is_72_l3487_348777

/-- Represents a rectangular strip with length and width -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular strip -/
def Strip.area (s : Strip) : ℝ := s.length * s.width

/-- Calculates the area of overlap between two perpendicular strips -/
def overlap_area (s : Strip) : ℝ := s.width * s.width

/-- The setup of the problem with four strips and overlaps -/
structure StripSetup where
  strips : Fin 4 → Strip
  num_overlaps : ℕ

/-- Theorem: The total area covered by four strips with given dimensions and overlaps is 72 -/
theorem total_covered_area_is_72 (setup : StripSetup) 
  (h1 : ∀ i, (setup.strips i).length = 12)
  (h2 : ∀ i, (setup.strips i).width = 2)
  (h3 : setup.num_overlaps = 6) :
  (Finset.sum Finset.univ (λ i => (setup.strips i).area)) - 
  (setup.num_overlaps : ℝ) * overlap_area (setup.strips 0) = 72 := by
  sorry


end total_covered_area_is_72_l3487_348777


namespace second_candidate_percentage_l3487_348705

/-- Represents an exam with total marks and passing marks. -/
structure Exam where
  totalMarks : ℝ
  passingMarks : ℝ

/-- Represents a candidate's performance in the exam. -/
structure Candidate where
  marksObtained : ℝ

def Exam.firstCandidateCondition (e : Exam) : Prop :=
  0.3 * e.totalMarks = e.passingMarks - 50

def Exam.secondCandidateCondition (e : Exam) (c : Candidate) : Prop :=
  c.marksObtained = e.passingMarks + 25

/-- The theorem stating the percentage of marks obtained by the second candidate. -/
theorem second_candidate_percentage (e : Exam) (c : Candidate) :
  e.passingMarks = 199.99999999999997 →
  e.firstCandidateCondition →
  e.secondCandidateCondition c →
  c.marksObtained / e.totalMarks = 0.45 := by
  sorry


end second_candidate_percentage_l3487_348705


namespace adult_ticket_cost_adult_ticket_cost_proof_l3487_348715

/-- The cost of an adult ticket for a play, given the following conditions:
  * Child tickets cost 1 dollar
  * 22 people attended the performance
  * Total ticket sales were 50 dollars
  * 18 children attended the play
-/
theorem adult_ticket_cost : ℝ → Prop :=
  fun adult_cost =>
    let child_cost : ℝ := 1
    let total_attendance : ℕ := 22
    let total_sales : ℝ := 50
    let children_attendance : ℕ := 18
    let adult_attendance : ℕ := total_attendance - children_attendance
    adult_cost * adult_attendance + child_cost * children_attendance = total_sales ∧
    adult_cost = 8

/-- Proof of the adult ticket cost theorem -/
theorem adult_ticket_cost_proof : ∃ (cost : ℝ), adult_ticket_cost cost := by
  sorry

end adult_ticket_cost_adult_ticket_cost_proof_l3487_348715


namespace pool_draining_and_filling_time_l3487_348704

/-- The time it takes for a pool to reach a certain water level when being simultaneously drained and filled -/
theorem pool_draining_and_filling_time 
  (pool_capacity : ℝ) 
  (drain_rate : ℝ) 
  (fill_rate : ℝ) 
  (final_volume : ℝ) 
  (h1 : pool_capacity = 120)
  (h2 : drain_rate = 1 / 4)
  (h3 : fill_rate = 1 / 6)
  (h4 : final_volume = 90) :
  ∃ t : ℝ, t = 3 ∧ 
  pool_capacity - (drain_rate * pool_capacity - fill_rate * pool_capacity) * t = final_volume :=
sorry

end pool_draining_and_filling_time_l3487_348704


namespace quadratic_roots_sum_bound_l3487_348744

theorem quadratic_roots_sum_bound (a b : ℤ) 
  (ha : a ≠ -1) (hb : b ≠ -1) 
  (h_roots : ∃ x y : ℤ, x ≠ y ∧ 
    x^2 + a*b*x + (a + b) = 0 ∧ 
    y^2 + a*b*y + (a + b) = 0) : 
  a + b ≤ 6 := by
sorry

end quadratic_roots_sum_bound_l3487_348744


namespace pure_imaginary_complex_number_l3487_348759

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) * (2 - Complex.I)
  (z.re = 0) → m = -2 := by
sorry

end pure_imaginary_complex_number_l3487_348759


namespace seed_mixture_percentage_l3487_348764

/-- Given two seed mixtures X and Y, and a final mixture composed of X and Y,
    this theorem proves the percentage of X in the final mixture. -/
theorem seed_mixture_percentage
  (x_ryegrass : Real) (x_bluegrass : Real)
  (y_ryegrass : Real) (y_fescue : Real)
  (final_ryegrass : Real) :
  x_ryegrass = 0.40 →
  x_bluegrass = 0.60 →
  y_ryegrass = 0.25 →
  y_fescue = 0.75 →
  final_ryegrass = 0.27 →
  x_ryegrass + x_bluegrass = 1 →
  y_ryegrass + y_fescue = 1 →
  ∃ (p : Real), p * x_ryegrass + (1 - p) * y_ryegrass = final_ryegrass ∧ p = 200 / 15 := by
  sorry

end seed_mixture_percentage_l3487_348764


namespace evaluate_expression_l3487_348719

theorem evaluate_expression : (1 / ((-5^3)^4)) * ((-5)^15) * (5^2) = -3125 := by
  sorry

end evaluate_expression_l3487_348719


namespace polynomial_square_prime_values_l3487_348714

def P (n : ℤ) : ℤ := n^3 - n^2 - 5*n + 2

theorem polynomial_square_prime_values :
  {n : ℤ | ∃ (p : ℕ), Prime p ∧ (P n)^2 = p^2} = {-3, -1, 0, 1, 3} := by
  sorry

end polynomial_square_prime_values_l3487_348714


namespace factorization_of_4_minus_4x_squared_l3487_348751

theorem factorization_of_4_minus_4x_squared (x : ℝ) : 4 - 4*x^2 = 4*(1+x)*(1-x) := by
  sorry

end factorization_of_4_minus_4x_squared_l3487_348751


namespace triangle_abc_properties_l3487_348722

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  -- Given condition
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos B →
  -- Conclusions
  B = π / 3 ∧
  (∀ x, x ∈ Set.Ioo (-3/2) (1/2) ↔
    ∃ A', 0 < A' ∧ A' < 2*π/3 ∧
    x = Real.sin A' * (Real.sqrt 3 * Real.cos A' - Real.sin A')) :=
by sorry

end triangle_abc_properties_l3487_348722


namespace problem_1_problem_2_problem_3_l3487_348799

-- Problem 1
theorem problem_1 : (-81) - (1/4 : ℚ) + (-7) - (3/4 : ℚ) - (-22) = -67 := by sorry

-- Problem 2
theorem problem_2 : -(4^2) / ((-2)^3) - 2^2 * (-1/2 : ℚ) = 4 := by sorry

-- Problem 3
theorem problem_3 : -(1^2023) - 24 * ((1/2 : ℚ) - 2/3 + 3/8) = -6 := by sorry

end problem_1_problem_2_problem_3_l3487_348799


namespace gcd_of_60_and_75_l3487_348758

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_of_60_and_75_l3487_348758


namespace problem_statement_l3487_348750

theorem problem_statement (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) :
  (1/3) * x^8 * y^9 = 2/5 := by sorry

end problem_statement_l3487_348750


namespace factorial_square_root_theorem_l3487_348726

theorem factorial_square_root_theorem : 
  (Real.sqrt ((Nat.factorial 5 * Nat.factorial 4) / Nat.factorial 3))^2 = 480 := by
  sorry

end factorial_square_root_theorem_l3487_348726


namespace print_shop_charge_difference_l3487_348746

/-- The charge difference for color copies between two print shops -/
theorem print_shop_charge_difference 
  (price_X : ℚ) -- Price per copy at shop X
  (price_Y : ℚ) -- Price per copy at shop Y
  (num_copies : ℕ) -- Number of copies
  (h1 : price_X = 1.25) -- Shop X charges $1.25 per copy
  (h2 : price_Y = 2.75) -- Shop Y charges $2.75 per copy
  (h3 : num_copies = 60) -- We're considering 60 copies
  : (price_Y - price_X) * num_copies = 90 := by
  sorry

#check print_shop_charge_difference

end print_shop_charge_difference_l3487_348746


namespace cookie_distribution_l3487_348768

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) :
  total_cookies = 24 →
  num_people = 6 →
  cookies_per_person = total_cookies / num_people →
  cookies_per_person = 4 :=
by sorry

end cookie_distribution_l3487_348768


namespace nested_root_equality_l3487_348709

theorem nested_root_equality (a : ℝ) (ha : a > 0) : 
  Real.sqrt (a * Real.sqrt (a * Real.sqrt a)) = a ^ (7/8) :=
by sorry

end nested_root_equality_l3487_348709


namespace complement_of_complement_l3487_348728

def V : Finset Nat := {1, 2, 3, 4, 5}

def C_VN : Finset Nat := {2, 4}

def N : Finset Nat := {1, 3, 5}

theorem complement_of_complement (V C_VN N : Finset Nat) 
  (hV : V = {1, 2, 3, 4, 5})
  (hC_VN : C_VN = {2, 4})
  (hN : N = {1, 3, 5}) :
  N = V \ C_VN :=
by sorry

end complement_of_complement_l3487_348728


namespace ball_max_height_l3487_348773

/-- The height function of the ball's trajectory -/
def f (t : ℝ) : ℝ := -10 * t^2 + 50 * t - 24

/-- The maximum height reached by the ball -/
def max_height : ℝ := 38.5

theorem ball_max_height :
  IsGreatest { y | ∃ t, f t = y } max_height := by
  sorry

end ball_max_height_l3487_348773


namespace no_polynomial_exists_l3487_348769

theorem no_polynomial_exists : ¬∃ (P : ℝ → ℝ → ℝ), 
  (∀ x y, x ∈ ({1, 2, 3} : Set ℝ) → y ∈ ({1, 2, 3} : Set ℝ) → 
    P x y ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 10} : Set ℝ)) ∧ 
  (∀ v, v ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 10} : Set ℝ) → 
    ∃! (x y : ℝ), x ∈ ({1, 2, 3} : Set ℝ) ∧ y ∈ ({1, 2, 3} : Set ℝ) ∧ P x y = v) ∧
  (∃ (a b c d e f : ℝ), ∀ x y, P x y = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f) :=
by sorry

end no_polynomial_exists_l3487_348769


namespace rhombus_diagonal_theorem_l3487_348767

/-- Represents a rhombus with given properties -/
structure Rhombus where
  diagonal1 : ℝ
  perimeter : ℝ
  diagonal2 : ℝ

/-- Theorem stating the relationship between the diagonals and perimeter of a rhombus -/
theorem rhombus_diagonal_theorem (r : Rhombus) (h1 : r.diagonal1 = 24) (h2 : r.perimeter = 52) :
  r.diagonal2 = 10 := by
  sorry

end rhombus_diagonal_theorem_l3487_348767


namespace squido_oysters_l3487_348712

theorem squido_oysters (squido crabby : ℕ) : 
  crabby ≥ 2 * squido →
  squido + crabby = 600 →
  squido = 200 :=
by
  sorry

end squido_oysters_l3487_348712


namespace complement_intersection_theorem_l3487_348723

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | |x| ≥ 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl A) ∩ (Set.compl B) = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end complement_intersection_theorem_l3487_348723


namespace pencil_distribution_l3487_348770

theorem pencil_distribution (initial_pencils : ℕ) (containers : ℕ) (additional_pencils : ℕ)
  (h1 : initial_pencils = 150)
  (h2 : containers = 5)
  (h3 : additional_pencils = 30) :
  (initial_pencils + additional_pencils) / containers = 36 :=
by sorry

end pencil_distribution_l3487_348770


namespace complex_division_result_l3487_348796

theorem complex_division_result : (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by
  sorry

end complex_division_result_l3487_348796


namespace area_of_specific_trapezoid_l3487_348710

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  /-- The length of the smaller segment of the larger lateral side -/
  smaller_segment : ℝ
  /-- The length of the larger segment of the larger lateral side -/
  larger_segment : ℝ
  /-- The smaller segment is positive -/
  smaller_segment_pos : 0 < smaller_segment
  /-- The larger segment is positive -/
  larger_segment_pos : 0 < larger_segment

/-- The area of a right trapezoid with an inscribed circle -/
def area (t : RightTrapezoidWithInscribedCircle) : ℝ :=
  18 -- Definition without proof

/-- Theorem stating that the area of the specific right trapezoid is 18 -/
theorem area_of_specific_trapezoid :
  ∀ t : RightTrapezoidWithInscribedCircle,
  t.smaller_segment = 1 ∧ t.larger_segment = 4 →
  area t = 18 := by
  sorry

end area_of_specific_trapezoid_l3487_348710


namespace students_without_portraits_l3487_348718

theorem students_without_portraits (total_students : ℕ) 
  (before_break : ℕ) (during_break : ℕ) (after_lunch : ℕ) : 
  total_students = 60 →
  before_break = total_students / 4 →
  during_break = (total_students - before_break) / 3 →
  after_lunch = 10 →
  total_students - (before_break + during_break + after_lunch) = 20 := by
sorry

end students_without_portraits_l3487_348718


namespace max_interval_increasing_sin_plus_cos_l3487_348761

theorem max_interval_increasing_sin_plus_cos :
  let f : ℝ → ℝ := λ x ↦ Real.sin x + Real.cos x
  ∃ a : ℝ, a = π / 4 ∧ 
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ a → f x < f y) ∧
    (∀ b : ℝ, b > a → ∃ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ b ∧ f x ≥ f y) :=
by sorry

end max_interval_increasing_sin_plus_cos_l3487_348761


namespace arithmetic_sequence_min_sum_l3487_348774

/-- An arithmetic sequence with common difference d, first term a₁, and sum function S_n -/
structure ArithmeticSequence where
  d : ℝ
  a₁ : ℝ
  S_n : ℕ → ℝ

/-- The sum of an arithmetic sequence reaches its minimum -/
def sum_reaches_minimum (seq : ArithmeticSequence) (n : ℕ) : Prop :=
  ∀ k : ℕ, seq.S_n k ≥ seq.S_n n

/-- Theorem: For an arithmetic sequence with non-zero common difference,
    negative first term, and S₇ = S₁₃, the sum reaches its minimum when n = 10 -/
theorem arithmetic_sequence_min_sum
  (seq : ArithmeticSequence)
  (h_d : seq.d ≠ 0)
  (h_a₁ : seq.a₁ < 0)
  (h_S : seq.S_n 7 = seq.S_n 13) :
  sum_reaches_minimum seq 10 := by
sorry

end arithmetic_sequence_min_sum_l3487_348774


namespace system_solution_exists_iff_l3487_348748

theorem system_solution_exists_iff (k : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - y = 4 ∧ k * x^2 + y = 5) ↔ k > -1/36 := by
  sorry

end system_solution_exists_iff_l3487_348748


namespace sqrt_equation_solution_l3487_348747

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 3) = 9 → x = 84 := by
  sorry

end sqrt_equation_solution_l3487_348747


namespace survey_b_count_l3487_348736

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => (firstSample + i * (populationSize / sampleSize)) % populationSize + 1)

/-- Count elements in a list that fall within a given range -/
def countInRange (list : List ℕ) (lower upper : ℕ) : ℕ :=
  list.filter (fun x => lower ≤ x ∧ x ≤ upper) |>.length

theorem survey_b_count :
  let populationSize := 480
  let sampleSize := 16
  let firstSample := 8
  let surveyBLower := 161
  let surveyBUpper := 320
  let sampledNumbers := systematicSample populationSize sampleSize firstSample
  countInRange sampledNumbers surveyBLower surveyBUpper = 5 := by
  sorry


end survey_b_count_l3487_348736


namespace soap_brand_survey_l3487_348753

theorem soap_brand_survey (total : ℕ) (only_A : ℕ) (both : ℕ) (only_B_ratio : ℕ) :
  total = 260 →
  only_A = 60 →
  both = 30 →
  only_B_ratio = 3 →
  total - (only_A + both + only_B_ratio * both) = 80 :=
by sorry

end soap_brand_survey_l3487_348753


namespace least_possible_area_of_square_l3487_348708

/-- The least possible side length of a square when measured as 7 cm to the nearest centimeter -/
def least_possible_side : ℝ := 6.5

/-- The measured side length of the square to the nearest centimeter -/
def measured_side : ℕ := 7

/-- The least possible area of the square -/
def least_possible_area : ℝ := least_possible_side ^ 2

theorem least_possible_area_of_square :
  least_possible_side ≥ (measured_side : ℝ) - 0.5 ∧
  least_possible_side < (measured_side : ℝ) ∧
  least_possible_area = 42.25 := by
  sorry

end least_possible_area_of_square_l3487_348708


namespace butterfat_mixture_proof_l3487_348741

-- Define the initial quantities and percentages
def initial_volume : ℝ := 8
def initial_butterfat_percentage : ℝ := 0.35
def added_butterfat_percentage : ℝ := 0.10
def target_butterfat_percentage : ℝ := 0.20

-- Define the volume of milk to be added
def added_volume : ℝ := 12

-- Theorem statement
theorem butterfat_mixture_proof :
  let total_volume := initial_volume + added_volume
  let total_butterfat := initial_volume * initial_butterfat_percentage + added_volume * added_butterfat_percentage
  total_butterfat / total_volume = target_butterfat_percentage := by
sorry


end butterfat_mixture_proof_l3487_348741


namespace coefficient_x_sqrt_x_in_expansion_l3487_348766

theorem coefficient_x_sqrt_x_in_expansion :
  let expansion := (λ x : ℝ => (Real.sqrt x - 1)^5)
  ∃ c : ℝ, ∀ x : ℝ, x > 0 →
    expansion x = c * x * Real.sqrt x + (λ y => y - c * x * Real.sqrt x) (expansion x) ∧
    c = 10 := by
  sorry

end coefficient_x_sqrt_x_in_expansion_l3487_348766


namespace cosine_function_parameters_l3487_348752

/-- Proves that for y = a cos(bx), if max is 3 at x=0 and first zero at x=π/6, then a=3 and b=3 -/
theorem cosine_function_parameters (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x, a * Real.cos (b * x) ≤ 3) → 
  a * Real.cos 0 = 3 → 
  a * Real.cos (b * (π / 6)) = 0 → 
  a = 3 ∧ b = 3 := by
  sorry


end cosine_function_parameters_l3487_348752


namespace c_work_time_l3487_348790

-- Define the work rates of a, b, and c
variable (A B C : ℝ)

-- Define the conditions
def condition1 : Prop := A + B = 1 / 6
def condition2 : Prop := B + C = 1 / 8
def condition3 : Prop := C + A = 1 / 12

-- Theorem statement
theorem c_work_time (h1 : condition1 A B) (h2 : condition2 B C) (h3 : condition3 C A) :
  1 / C = 48 := by sorry

end c_work_time_l3487_348790


namespace chocolate_bars_in_large_box_l3487_348734

/-- The number of chocolate bars in the large box -/
def total_chocolate_bars (num_small_boxes : ℕ) (bars_per_small_box : ℕ) : ℕ :=
  num_small_boxes * bars_per_small_box

/-- Theorem stating the total number of chocolate bars in the large box -/
theorem chocolate_bars_in_large_box :
  total_chocolate_bars 20 32 = 640 := by
  sorry

end chocolate_bars_in_large_box_l3487_348734


namespace square_area_from_perimeter_l3487_348781

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 20) :
  let side_length := perimeter / 4
  let area := side_length * side_length
  area = 25 := by sorry

end square_area_from_perimeter_l3487_348781


namespace root_sum_of_coefficients_l3487_348789

theorem root_sum_of_coefficients (a b : ℝ) : 
  (Complex.I * 2 + 1) ^ 2 + a * (Complex.I * 2 + 1) + b = 0 → a + b = 3 := by
  sorry

end root_sum_of_coefficients_l3487_348789


namespace distance_between_signs_l3487_348795

theorem distance_between_signs 
  (total_distance : ℕ) 
  (distance_to_first_sign : ℕ) 
  (distance_after_second_sign : ℕ) 
  (h1 : total_distance = 1000)
  (h2 : distance_to_first_sign = 350)
  (h3 : distance_after_second_sign = 275) :
  total_distance - distance_to_first_sign - distance_after_second_sign = 375 :=
by sorry

end distance_between_signs_l3487_348795


namespace negative_among_expressions_l3487_348756

theorem negative_among_expressions : 
  (|(-3)| > 0) ∧ (-(-3) > 0) ∧ ((-3)^2 > 0) ∧ (-Real.sqrt 3 < 0) := by
  sorry

end negative_among_expressions_l3487_348756


namespace no_square_base_l3487_348785

theorem no_square_base (b : ℕ) (h : b > 0) : ¬∃ (n : ℕ), b^2 + 3*b + 2 = n^2 := by
  sorry

end no_square_base_l3487_348785


namespace abs_equation_solution_difference_l3487_348779

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ + 5| = 20) ∧ (|x₂ + 5| = 20) ∧ (x₁ ≠ x₂) ∧ (|x₁ - x₂| = 40) := by
  sorry

end abs_equation_solution_difference_l3487_348779


namespace divide_subtract_problem_l3487_348760

theorem divide_subtract_problem (x : ℝ) : 
  (990 / x) - 100 = 10 → x = 9 := by sorry

end divide_subtract_problem_l3487_348760


namespace simple_interest_rate_calculation_l3487_348788

theorem simple_interest_rate_calculation (total_amount interest_difference amount_B : ℚ)
  (h1 : total_amount = 10000)
  (h2 : interest_difference = 360)
  (h3 : amount_B = 4000) :
  let amount_A := total_amount - amount_B
  let rate_A := 15 / 100
  let time := 2
  let interest_A := amount_A * rate_A * time
  let interest_B := interest_A - interest_difference
  let rate_B := interest_B / (amount_B * time)
  rate_B = 18 / 100 := by sorry

end simple_interest_rate_calculation_l3487_348788


namespace count_numbers_with_digits_eq_six_l3487_348707

/-- The count of integers between 600 and 2000 that contain the digits 3, 5, and 7 -/
def count_numbers_with_digits : ℕ :=
  -- Definition goes here
  sorry

/-- The range of integers to consider -/
def lower_bound : ℕ := 600
def upper_bound : ℕ := 2000

/-- The required digits -/
def required_digits : List ℕ := [3, 5, 7]

theorem count_numbers_with_digits_eq_six :
  count_numbers_with_digits = 6 :=
sorry

end count_numbers_with_digits_eq_six_l3487_348707


namespace ball_selection_limit_l3487_348776

open Real

/-- The probability of selecting n₁ white balls and n₂ black balls without replacement from an urn -/
noncomputable def P (M₁ M₂ n₁ n₂ : ℕ) : ℝ :=
  (Nat.choose M₁ n₁ * Nat.choose M₂ n₂ : ℝ) / Nat.choose (M₁ + M₂) (n₁ + n₂)

/-- The limit of the probability as M and M₁ approach infinity -/
theorem ball_selection_limit (n₁ n₂ : ℕ) (p : ℝ) (h_p : 0 < p ∧ p < 1) :
  ∀ ε > 0, ∃ N : ℕ, ∀ M₁ M₂ : ℕ,
    M₁ ≥ N → M₂ ≥ N →
    |P M₁ M₂ n₁ n₂ - (Nat.choose (n₁ + n₂) n₁ : ℝ) * p^n₁ * (1 - p)^n₂| < ε :=
by sorry

end ball_selection_limit_l3487_348776


namespace km2_to_hectares_conversion_m2_to_km2_conversion_l3487_348706

-- Define the conversion factors
def km2_to_hectares : ℝ := 100
def m2_to_km2 : ℝ := 1000000

-- Theorem 1: 3.4 km² = 340 hectares
theorem km2_to_hectares_conversion :
  3.4 * km2_to_hectares = 340 := by sorry

-- Theorem 2: 690000 m² = 0.69 km²
theorem m2_to_km2_conversion :
  690000 / m2_to_km2 = 0.69 := by sorry

end km2_to_hectares_conversion_m2_to_km2_conversion_l3487_348706


namespace perfect_cubes_with_special_property_l3487_348775

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def last_three_digits (n : ℕ) : ℕ := n % 1000

def erase_last_three_digits (n : ℕ) : ℕ := n / 1000

theorem perfect_cubes_with_special_property :
  ∀ n : ℕ,
    n > 0 ∧
    is_perfect_cube n ∧
    n % 10 ≠ 0 ∧
    is_perfect_cube (erase_last_three_digits n) →
    n = 1331 ∨ n = 1728 :=
by sorry

end perfect_cubes_with_special_property_l3487_348775


namespace weight_of_ten_moles_C6H8O6_l3487_348772

/-- The weight of 10 moles of C6H8O6 -/
theorem weight_of_ten_moles_C6H8O6 
  (atomic_weight_C : ℝ) 
  (atomic_weight_H : ℝ) 
  (atomic_weight_O : ℝ) 
  (h1 : atomic_weight_C = 12.01)
  (h2 : atomic_weight_H = 1.008)
  (h3 : atomic_weight_O = 16.00) : 
  10 * (6 * atomic_weight_C + 8 * atomic_weight_H + 6 * atomic_weight_O) = 1761.24 := by
sorry

end weight_of_ten_moles_C6H8O6_l3487_348772


namespace inequality_multiplication_l3487_348701

theorem inequality_multiplication (x y : ℝ) (h : x > y) : 3 * x > 3 * y := by
  sorry

end inequality_multiplication_l3487_348701


namespace angle_T_measure_l3487_348757

-- Define a pentagon
structure Pentagon where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ

-- Define the properties of the pentagon
def is_valid_pentagon (p : Pentagon) : Prop :=
  p.P + p.Q + p.R + p.S + p.T = 540

def angles_congruent (p : Pentagon) : Prop :=
  p.P = p.R ∧ p.R = p.T

def angles_supplementary (p : Pentagon) : Prop :=
  p.Q + p.S = 180

-- Theorem statement
theorem angle_T_measure (p : Pentagon) 
  (h1 : is_valid_pentagon p) 
  (h2 : angles_congruent p) 
  (h3 : angles_supplementary p) : 
  p.T = 120 := by sorry

end angle_T_measure_l3487_348757


namespace non_shaded_perimeter_l3487_348784

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (large : Rectangle) (small : Rectangle) :
  large.width = 12 ∧ 
  large.height = 12 ∧ 
  small.width = 6 ∧ 
  small.height = 4 ∧ 
  large.area - small.area = 144 →
  Rectangle.perimeter { width := large.width - small.width, height := large.height } +
  Rectangle.perimeter { width := large.width, height := large.height - small.height } = 28 := by
  sorry

end non_shaded_perimeter_l3487_348784


namespace parabola_kite_sum_l3487_348727

/-- Given two parabolas that form a kite when intersecting the coordinate axes, 
    prove that the sum of their coefficients is 1/50 if the kite area is 20 -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    (a * x₁^2 + 3 = 0 ∧ 5 - b * x₁^2 = 0) ∧ 
    (a * x₂^2 + 3 = 0 ∧ 5 - b * x₂^2 = 0) ∧
    (y₁ = a * 0^2 + 3 ∧ y₂ = 5 - b * 0^2) ∧
    (1/2 * (x₂ - x₁) * (y₂ - y₁) = 20)) →
  a + b = 1/50 := by
sorry

end parabola_kite_sum_l3487_348727


namespace nth_S_645_l3487_348729

/-- The set of positive integers with remainder 5 when divided by 8 -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ n % 8 = 5}

/-- The nth element of S -/
def nth_S (n : ℕ) : ℕ := 8 * (n - 1) + 5

theorem nth_S_645 : nth_S 81 = 645 := by
  sorry

end nth_S_645_l3487_348729


namespace problem_solution_l3487_348782

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) : x^7 - 6*x^5 + 5*x^3 - x = 0 := by
  sorry

end problem_solution_l3487_348782


namespace complex_power_modulus_l3487_348755

theorem complex_power_modulus : Complex.abs ((2 + Complex.I) ^ 8) = 625 := by
  sorry

end complex_power_modulus_l3487_348755


namespace inequality_proof_l3487_348720

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end inequality_proof_l3487_348720


namespace first_group_size_l3487_348735

def work_rate (people : ℕ) (time : ℕ) : ℚ := 1 / (people * time)

theorem first_group_size :
  ∀ (p : ℕ),
  (work_rate p 60 = work_rate 16 30) →
  p = 8 := by
sorry

end first_group_size_l3487_348735


namespace simplify_and_evaluate_l3487_348711

theorem simplify_and_evaluate (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  5 * (3 * x^2 * y - x * y^2) - (x * y^2 + 3 * x^2 * y) = 36 := by
  sorry

end simplify_and_evaluate_l3487_348711


namespace approximation_accuracy_l3487_348786

theorem approximation_accuracy : 
  abs (84 * Real.sqrt 7 - 222 * (2 / (2 + 7))) < 0.001 := by
  sorry

end approximation_accuracy_l3487_348786


namespace emerald_density_conversion_l3487_348780

/-- Density of a material in g/cm³ -/
def density : ℝ := 2.7

/-- Conversion factor from grams to carats -/
def gramsToCarat : ℝ := 5

/-- Conversion factor from cubic centimeters to cubic inches -/
def cmCubedToInchCubed : ℝ := 16.387

/-- Density of emerald in carats per cubic inch -/
def emeraldDensityCaratsPerCubicInch : ℝ :=
  density * gramsToCarat * cmCubedToInchCubed

theorem emerald_density_conversion :
  ⌊emeraldDensityCaratsPerCubicInch⌋ = 221 := by sorry

end emerald_density_conversion_l3487_348780


namespace parabola_vertex_l3487_348798

/-- The vertex of the parabola y = x^2 - 2x + 4 has coordinates (1, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = x^2 - 2*x + 4 → (∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ 
    ∀ (x' : ℝ), x'^2 - 2*x' + 4 ≥ k ∧ 
    (x'^2 - 2*x' + 4 = k ↔ x' = h)) := by
  sorry

end parabola_vertex_l3487_348798


namespace not_diff_of_squares_l3487_348763

theorem not_diff_of_squares (a : ℤ) : ¬ ∃ (x y : ℤ), 4 * a + 2 = x^2 - y^2 := by
  sorry

end not_diff_of_squares_l3487_348763


namespace number_equation_l3487_348700

theorem number_equation (x n : ℝ) : 
  x = 596.95 → 3639 + n - x = 3054 → n = 11.95 := by sorry

end number_equation_l3487_348700


namespace sin_20_sqrt3_plus_tan_50_equals_one_l3487_348731

theorem sin_20_sqrt3_plus_tan_50_equals_one :
  Real.sin (20 * π / 180) * (Real.sqrt 3 + Real.tan (50 * π / 180)) = 1 := by
  sorry

end sin_20_sqrt3_plus_tan_50_equals_one_l3487_348731


namespace cid_oil_changes_l3487_348724

/-- Represents the mechanic shop's pricing and services -/
structure MechanicShop where
  oil_change_price : ℕ
  repair_price : ℕ
  car_wash_price : ℕ
  repaired_cars : ℕ
  washed_cars : ℕ
  total_earnings : ℕ

/-- Calculates the number of oil changes given the shop's data -/
def calculate_oil_changes (shop : MechanicShop) : ℕ :=
  (shop.total_earnings - shop.repair_price * shop.repaired_cars - shop.car_wash_price * shop.washed_cars) / shop.oil_change_price

/-- Theorem stating that Cid changed the oil for 5 cars -/
theorem cid_oil_changes :
  let shop : MechanicShop := {
    oil_change_price := 20,
    repair_price := 30,
    car_wash_price := 5,
    repaired_cars := 10,
    washed_cars := 15,
    total_earnings := 475
  }
  calculate_oil_changes shop = 5 := by sorry

end cid_oil_changes_l3487_348724


namespace abs_eq_neg_self_iff_nonpositive_l3487_348754

theorem abs_eq_neg_self_iff_nonpositive (x : ℝ) : |x| = -x ↔ x ≤ 0 := by sorry

end abs_eq_neg_self_iff_nonpositive_l3487_348754


namespace trig_expression_equality_l3487_348737

theorem trig_expression_equality : 
  (Real.sin (110 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (155 * π / 180)^2 - Real.sin (155 * π / 180)^2) = 1/2 := by
  sorry

end trig_expression_equality_l3487_348737


namespace smallest_n_value_l3487_348733

theorem smallest_n_value (r g b : ℕ) (n : ℕ) : 
  (∃ m : ℕ, m > 0 ∧ 10 * r = m ∧ 18 * g = m ∧ 24 * b = m ∧ 25 * n = m) →
  n ≥ 15 :=
by sorry

end smallest_n_value_l3487_348733


namespace smallest_a_value_l3487_348730

theorem smallest_a_value (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0)
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (15 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (15 * ↑x)) → a' ≥ 15 :=
by sorry

end smallest_a_value_l3487_348730


namespace wheat_flour_amount_l3487_348717

/-- The amount of wheat flour used by the bakery -/
def wheat_flour : ℝ := sorry

/-- The amount of white flour used by the bakery -/
def white_flour : ℝ := 0.1

/-- The total amount of flour used by the bakery -/
def total_flour : ℝ := 0.3

/-- Theorem stating that the amount of wheat flour used is 0.2 bags -/
theorem wheat_flour_amount : wheat_flour = 0.2 := by
  sorry

end wheat_flour_amount_l3487_348717


namespace circplus_neg_three_eight_l3487_348783

/-- The ⊕ operation for rational numbers -/
def circplus (a b : ℚ) : ℚ := a * b + (a - b)

/-- Theorem stating that (-3) ⊕ 8 = -35 -/
theorem circplus_neg_three_eight : circplus (-3) 8 = -35 := by sorry

end circplus_neg_three_eight_l3487_348783


namespace kim_morning_routine_time_l3487_348745

/-- Represents the types of employees in Kim's office. -/
inductive EmployeeType
  | Senior
  | Junior
  | Intern

/-- Represents whether an employee worked overtime or not. -/
inductive OvertimeStatus
  | Overtime
  | NoOvertime

/-- Calculates the total time for Kim's morning routine. -/
def morning_routine_time (
  senior_count junior_count intern_count : Nat
  ) (
  senior_overtime junior_overtime intern_overtime : Nat
  ) : Nat :=
  let coffee_time := 5
  let status_update_time :=
    3 * senior_count + 2 * junior_count + 1 * intern_count
  let payroll_update_time :=
    4 * senior_overtime + 2 * (senior_count - senior_overtime) +
    3 * junior_overtime + 1 * (junior_count - junior_overtime) +
    2 * intern_overtime + 1 -- 1 minute for 2 interns without overtime (30 seconds each)
  let task_allocation_time :=
    4 * senior_count + 3 * junior_count + 2 * intern_count
  let additional_tasks_time := 10 + 8 + 6 + 5

  coffee_time + status_update_time + payroll_update_time +
  task_allocation_time + additional_tasks_time

/-- Theorem stating that Kim's morning routine takes 101 minutes. -/
theorem kim_morning_routine_time :
  morning_routine_time 3 3 3 2 3 1 = 101 := by
  sorry

end kim_morning_routine_time_l3487_348745


namespace equal_roots_condition_l3487_348743

theorem equal_roots_condition (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2*x - (m^2 + 2)) / ((x^2 - 2)*(m - 2)) = x / m) → 
  m = -2 := by
  sorry

end equal_roots_condition_l3487_348743


namespace fifth_smallest_odd_with_four_prime_factors_l3487_348778

def has_at_least_four_prime_factors (n : ℕ) : Prop :=
  ∃ (p q r s : ℕ), Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧ p * q * r * s ∣ n

def is_fifth_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), P a ∧ P b ∧ P c ∧ P d ∧
    a < b ∧ b < c ∧ c < d ∧ d < n ∧
    (∀ m, P m → m ≥ n ∨ m = a ∨ m = b ∨ m = c ∨ m = d)

theorem fifth_smallest_odd_with_four_prime_factors :
  is_fifth_smallest (λ n => Odd n ∧ has_at_least_four_prime_factors n) 1925 :=
sorry

end fifth_smallest_odd_with_four_prime_factors_l3487_348778


namespace cube_difference_divided_problem_solution_l3487_348791

theorem cube_difference_divided (a b : ℕ) (h : a > b) :
  (a^3 - b^3) / (a - b) = a^2 + a*b + b^2 :=
by sorry

theorem problem_solution : (64^3 - 27^3) / 37 = 6553 :=
by
  have h : 64 > 27 := by sorry
  have := cube_difference_divided 64 27 h
  sorry

end cube_difference_divided_problem_solution_l3487_348791


namespace max_value_of_expression_l3487_348787

theorem max_value_of_expression (x y z : ℝ) 
  (h : 2 * x^2 + y^2 + z^2 = 2 * x - 4 * y + 2 * x * z - 5) : 
  ∃ (M : ℝ), M = 4 ∧ ∀ (a b c : ℝ), 2 * a^2 + b^2 + c^2 = 2 * a - 4 * b + 2 * a * c - 5 → 
  a - b + c ≤ M :=
by sorry

end max_value_of_expression_l3487_348787


namespace existence_of_no_seven_multiple_l3487_348792

/-- Function to check if a natural number contains the digit 7 in its decimal representation -/
def containsSeven (n : ℕ) : Prop := sorry

/-- Function to generate the sequence of numbers obtained by multiplying by 5 k times -/
def multiplyByFive (n : ℕ) (k : ℕ) : List ℕ := sorry

/-- Function to generate the sequence of numbers obtained by multiplying by 2 k times -/
def multiplyByTwo (n : ℕ) (k : ℕ) : List ℕ := sorry

/-- Theorem stating the existence of a number that can be multiplied by 2 k times
    without producing a number containing 7, given a number that can be multiplied
    by 5 k times without producing a number containing 7 -/
theorem existence_of_no_seven_multiple (n : ℕ) (k : ℕ) :
  (∀ m ∈ multiplyByFive n k, ¬containsSeven m) →
  ∃ m : ℕ, ∀ p ∈ multiplyByTwo m k, ¬containsSeven p :=
by sorry

end existence_of_no_seven_multiple_l3487_348792


namespace intersection_implies_a_value_l3487_348716

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a+1, a^2+3}

-- State the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A a ∩ B a = {-3}) → a = -2 := by
  sorry

end intersection_implies_a_value_l3487_348716
