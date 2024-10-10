import Mathlib

namespace linear_equation_solution_l2135_213564

theorem linear_equation_solution (b : ℝ) : 
  (∀ x y : ℝ, x - 2*y + b = 0 → y = (1/2)*x + b - 1) → b = 2 := by
  sorry

end linear_equation_solution_l2135_213564


namespace nested_fraction_equality_l2135_213568

theorem nested_fraction_equality : 
  1 + 1 / (1 - 1 / (2 + 1 / 3)) = 11 / 4 := by sorry

end nested_fraction_equality_l2135_213568


namespace largest_divisor_of_difference_of_squares_l2135_213583

theorem largest_divisor_of_difference_of_squares (m n : ℤ) : 
  Odd m → Odd n → n < m → 
  (∀ k : ℤ, k ∣ (m^2 - n^2) → k ≤ 8) ∧ 8 ∣ (m^2 - n^2) :=
sorry

end largest_divisor_of_difference_of_squares_l2135_213583


namespace modulo_equivalence_unique_solution_l2135_213509

theorem modulo_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2839 [ZMOD 10] ∧ n = 1 := by sorry

end modulo_equivalence_unique_solution_l2135_213509


namespace vector_angle_problem_l2135_213579

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_angle_problem (a b : ℝ × ℝ) 
  (h1 : a.1^2 + a.2^2 = 4)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 2)
  (h3 : (a.1 + b.1) * (3 * a.1 - b.1) + (a.2 + b.2) * (3 * a.2 - b.2) = 4) :
  angle_between_vectors a b = 2 * Real.pi / 3 := by
  sorry

end vector_angle_problem_l2135_213579


namespace sin_585_degrees_l2135_213535

theorem sin_585_degrees : Real.sin (585 * π / 180) = - Real.sqrt 2 / 2 := by
  sorry

end sin_585_degrees_l2135_213535


namespace expression_not_equal_one_l2135_213584

theorem expression_not_equal_one (a y : ℝ) (ha : a ≠ 0) (hay : a ≠ y) :
  (a / (a - y) + y / (a + y)) / (y / (a - y) - a / (a + y)) ≠ 1 :=
by sorry

end expression_not_equal_one_l2135_213584


namespace library_books_l2135_213532

theorem library_books (shelves : ℝ) (books_per_shelf : ℕ) : 
  shelves = 14240.0 → books_per_shelf = 8 → shelves * (books_per_shelf : ℝ) = 113920 := by
  sorry

end library_books_l2135_213532


namespace tom_has_two_yellow_tickets_l2135_213582

/-- Represents the number of tickets Tom has -/
structure TicketHoldings where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Conversion rates between ticket types -/
def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10

/-- The number of additional blue tickets Tom needs -/
def additional_blue_needed : ℕ := 163

/-- Tom's current ticket holdings -/
def toms_tickets : TicketHoldings := {
  yellow := 0,  -- We don't know this value yet, so we set it to 0
  red := 3,
  blue := 7
}

/-- Theorem stating that Tom has 2 yellow tickets -/
theorem tom_has_two_yellow_tickets :
  ∃ (y : ℕ), 
    y * (yellow_to_red * red_to_blue) + 
    toms_tickets.red * red_to_blue + 
    toms_tickets.blue + 
    additional_blue_needed = 
    2 * (yellow_to_red * red_to_blue) ∧
    y = 2 := by
  sorry


end tom_has_two_yellow_tickets_l2135_213582


namespace smallest_c_for_no_five_l2135_213518

theorem smallest_c_for_no_five : ∃ c : ℤ, (∀ x : ℝ, x^2 + c*x + 10 ≠ 5) ∧ 
  (∀ c' : ℤ, c' < c → ∃ x : ℝ, x^2 + c'*x + 10 = 5) :=
by sorry

end smallest_c_for_no_five_l2135_213518


namespace distribute_negative_two_l2135_213563

theorem distribute_negative_two (m n : ℝ) : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end distribute_negative_two_l2135_213563


namespace distribute_students_count_l2135_213587

/-- The number of ways to distribute four students into three classes -/
def distribute_students : ℕ :=
  let total_distributions := (4 : ℕ).choose 2 * (3 : ℕ).factorial
  let invalid_distributions := (3 : ℕ).factorial
  total_distributions - invalid_distributions

/-- Theorem stating that the number of valid distributions is 30 -/
theorem distribute_students_count : distribute_students = 30 := by
  sorry

end distribute_students_count_l2135_213587


namespace tank_insulation_problem_l2135_213505

theorem tank_insulation_problem (x : ℝ) : 
  x > 0 →  -- Ensure x is positive
  (14 * x + 20) * 20 = 1520 → 
  x = 4 := by
sorry

end tank_insulation_problem_l2135_213505


namespace only_white_balls_drawn_is_random_variable_l2135_213570

/-- A bag containing white and red balls -/
structure Bag where
  white_balls : ℕ
  red_balls : ℕ

/-- The options for potential random variables -/
inductive DrawOption
  | BallsDrawn
  | WhiteBallsDrawn
  | TotalBallsDrawn
  | TotalBallsInBag

/-- Definition of a random variable in this context -/
def is_random_variable (option : DrawOption) (bag : Bag) (num_drawn : ℕ) : Prop :=
  match option with
  | DrawOption.BallsDrawn => num_drawn ≠ num_drawn
  | DrawOption.WhiteBallsDrawn => true
  | DrawOption.TotalBallsDrawn => num_drawn ≠ num_drawn
  | DrawOption.TotalBallsInBag => bag.white_balls + bag.red_balls ≠ bag.white_balls + bag.red_balls

/-- The main theorem stating that only the number of white balls drawn is a random variable -/
theorem only_white_balls_drawn_is_random_variable (bag : Bag) (num_drawn : ℕ) :
  bag.white_balls = 5 → bag.red_balls = 3 → num_drawn = 3 →
  ∀ (option : DrawOption), is_random_variable option bag num_drawn ↔ option = DrawOption.WhiteBallsDrawn :=
by sorry

end only_white_balls_drawn_is_random_variable_l2135_213570


namespace polynomial_multiplication_simplification_l2135_213540

theorem polynomial_multiplication_simplification (x : ℝ) :
  (3*x - 2) * (5*x^12 - 3*x^11 + 4*x^9 - 2*x^8) =
  15*x^13 - 19*x^12 + 6*x^11 + 12*x^10 - 14*x^9 - 4*x^8 := by
  sorry

end polynomial_multiplication_simplification_l2135_213540


namespace rectangle_diagonal_l2135_213581

theorem rectangle_diagonal (l w : ℝ) (h_area : l * w = 20) (h_perimeter : 2 * l + 2 * w = 18) :
  l^2 + w^2 = 41 := by
  sorry

end rectangle_diagonal_l2135_213581


namespace emir_needs_two_more_dollars_l2135_213544

/-- The amount of additional money Emir needs to buy three books --/
def additional_money_needed (dictionary_cost cookbook_cost dinosaur_book_cost savings : ℕ) : ℕ :=
  (dictionary_cost + cookbook_cost + dinosaur_book_cost) - savings

/-- Theorem: Emir needs $2 more to buy all three books --/
theorem emir_needs_two_more_dollars : 
  additional_money_needed 5 5 11 19 = 2 := by
  sorry

end emir_needs_two_more_dollars_l2135_213544


namespace distribute_three_items_five_people_l2135_213538

/-- The number of ways to distribute distinct items among distinct people -/
def distribute_items (num_items : ℕ) (num_people : ℕ) : ℕ :=
  num_people ^ num_items

/-- Theorem: Distributing 3 distinct items among 5 distinct people results in 125 ways -/
theorem distribute_three_items_five_people : 
  distribute_items 3 5 = 125 := by
  sorry

end distribute_three_items_five_people_l2135_213538


namespace base_10_number_l2135_213507

-- Define the properties of the number
def is_valid_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a < 8 ∧ b < 8 ∧ c < 8 ∧ d < 8 ∧
  (8 * a + b + c / 8 + d / 64 : ℚ) = (12 * b + b + b / 12 + a / 144 : ℚ)

-- State the theorem
theorem base_10_number (a b c d : ℕ) :
  is_valid_number a b c d → a * 100 + b * 10 + c = 321 :=
by sorry

end base_10_number_l2135_213507


namespace range_of_m_l2135_213593

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < m^2 - m) → m < -1 ∨ m > 2 := by
  sorry

end range_of_m_l2135_213593


namespace matrix_equation_l2135_213575

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, 10; 8, -4]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![10/7, -40/7; -4/7, 16/7]

theorem matrix_equation : N * A = B := by sorry

end matrix_equation_l2135_213575


namespace parabola_intersection_theorem_l2135_213504

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * x - Real.sqrt 3

-- Define point M
def point_M : ℝ × ℝ := (1, 0)

-- Define the midpoint condition
def is_midpoint (m a b : ℝ × ℝ) : Prop :=
  m.1 = (a.1 + b.1) / 2 ∧ m.2 = (a.2 + b.2) / 2

-- Main theorem
theorem parabola_intersection_theorem :
  ∃ (p : ℝ) (a b : ℝ × ℝ),
    parabola p b.1 b.2 ∧
    line b.1 b.2 ∧
    is_midpoint point_M a b →
    p = 2 :=
sorry

end parabola_intersection_theorem_l2135_213504


namespace solve_for_T_l2135_213578

theorem solve_for_T : ∃ T : ℚ, (3/4) * (1/6) * T = (1/5) * (1/4) * 120 ∧ T = 48 := by
  sorry

end solve_for_T_l2135_213578


namespace A_equals_2B_l2135_213586

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x - 2 * B^2
def g (B x : ℝ) : ℝ := B * x

-- State the theorem
theorem A_equals_2B (A B : ℝ) (h1 : B ≠ 0) (h2 : f A B (g B 1) = 0) : A = 2 * B := by
  sorry

end A_equals_2B_l2135_213586


namespace angle_between_clock_hands_at_8_30_angle_between_clock_hands_at_8_30_is_75_l2135_213585

/-- The angle between clock hands at 8:30 --/
theorem angle_between_clock_hands_at_8_30 : ℝ :=
  let hours : ℝ := 8.5
  let minutes : ℝ := 30
  let angle_per_hour : ℝ := 360 / 12
  let hour_hand_angle : ℝ := hours * angle_per_hour
  let minute_hand_angle : ℝ := minutes * (360 / 60)
  let angle_diff : ℝ := |hour_hand_angle - minute_hand_angle|
  75

/-- Proof that the angle between clock hands at 8:30 is 75° --/
theorem angle_between_clock_hands_at_8_30_is_75 :
  angle_between_clock_hands_at_8_30 = 75 := by
  sorry

end angle_between_clock_hands_at_8_30_angle_between_clock_hands_at_8_30_is_75_l2135_213585


namespace symmetric_difference_of_A_and_B_l2135_213512

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ -9/4}
def B : Set ℝ := {x | x < 0}

-- Define set difference
def setDifference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define symmetric difference
def symmetricDifference (M N : Set ℝ) : Set ℝ := 
  (setDifference M N) ∪ (setDifference N M)

-- Theorem statement
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {x | x ≥ 0 ∨ x < -9/4} := by
  sorry

end symmetric_difference_of_A_and_B_l2135_213512


namespace max_sock_price_l2135_213555

theorem max_sock_price (total_money : ℕ) (entrance_fee : ℕ) (num_socks : ℕ) (tax_rate : ℚ) :
  total_money = 180 →
  entrance_fee = 3 →
  num_socks = 20 →
  tax_rate = 6 / 100 →
  ∃ (max_price : ℕ), 
    max_price = 8 ∧
    (max_price : ℚ) * num_socks * (1 + tax_rate) + entrance_fee ≤ total_money ∧
    ∀ (price : ℕ), 
      price > max_price → 
      (price : ℚ) * num_socks * (1 + tax_rate) + entrance_fee > total_money :=
by sorry

end max_sock_price_l2135_213555


namespace fraction_integer_iff_p_in_set_l2135_213580

theorem fraction_integer_iff_p_in_set (p : ℕ) (hp : p > 0) :
  (∃ k : ℕ, k > 0 ∧ (4 * p + 34 : ℚ) / (3 * p - 8 : ℚ) = k) ↔ p ∈ ({3, 4, 5, 12} : Set ℕ) :=
sorry

end fraction_integer_iff_p_in_set_l2135_213580


namespace floor_painting_dimensions_l2135_213569

theorem floor_painting_dimensions :
  ∀ (a b x : ℕ),
  0 < a → 0 < b →
  b > a →
  a + b = 15 →
  (a - 2*x) * (b - 2*x) = 2 * a * b / 3 →
  (a = 8 ∧ b = 7) ∨ (a = 7 ∧ b = 8) :=
by sorry

end floor_painting_dimensions_l2135_213569


namespace fraction_subtraction_simplification_l2135_213519

theorem fraction_subtraction_simplification : 
  (9 : ℚ) / 19 - 5 / 57 - 2 / 38 = 1 / 3 := by
  sorry

end fraction_subtraction_simplification_l2135_213519


namespace smallest_prime_divisor_of_sum_l2135_213531

theorem smallest_prime_divisor_of_sum (n : ℕ) (m : ℕ) :
  2 = Nat.minFac (3^25 + 11^19) := by
  sorry

end smallest_prime_divisor_of_sum_l2135_213531


namespace infinite_geometric_series_first_term_l2135_213571

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 12) 
  (h3 : S = a / (1 - r)) : 
  a = 16 := by
sorry

end infinite_geometric_series_first_term_l2135_213571


namespace root_squares_sum_l2135_213542

theorem root_squares_sum (a b : ℝ) (a_ne_b : a ≠ b) : 
  ∃ (s t : ℝ), (a * s^2 + b * s + b = 0) ∧ 
                (a * t^2 + a * t + b = 0) ∧ 
                (s * t = 1) → 
                (s^2 + t^2 = 3) := by
  sorry

end root_squares_sum_l2135_213542


namespace elvins_internet_charge_l2135_213573

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ
  totalBill : ℝ
  totalBill_eq : totalBill = callCharge + internetCharge

/-- Theorem stating Elvin's fixed monthly internet charge -/
theorem elvins_internet_charge 
  (jan : MonthlyBill) 
  (feb : MonthlyBill) 
  (h1 : jan.totalBill = 40)
  (h2 : feb.totalBill = 76)
  (h3 : feb.callCharge = 2 * jan.callCharge)
  (h4 : jan.internetCharge = feb.internetCharge) : 
  jan.internetCharge = 4 := by
sorry

end elvins_internet_charge_l2135_213573


namespace tan_C_minus_pi_4_max_area_l2135_213560

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.a * t.b + t.c^2

-- Part I
theorem tan_C_minus_pi_4 (t : Triangle) (h : satisfiesCondition t) :
  Real.tan (t.C - π/4) = 2 - Real.sqrt 3 := by
  sorry

-- Part II
theorem max_area (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.c = Real.sqrt 3) :
  (∀ s : Triangle, satisfiesCondition s → s.c = Real.sqrt 3 →
    t.a * t.b * Real.sin t.C / 2 ≥ s.a * s.b * Real.sin s.C / 2) →
  t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 4 := by
  sorry

end tan_C_minus_pi_4_max_area_l2135_213560


namespace find_divisor_l2135_213536

theorem find_divisor (dividend quotient remainder : ℕ) : 
  dividend = 12401 → quotient = 76 → remainder = 13 →
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 163 := by
  sorry

end find_divisor_l2135_213536


namespace camel_cost_l2135_213553

-- Define the cost of each animal as a real number
variable (camel horse ox elephant lion bear : ℝ)

-- Define the relationships between animal costs
axiom camel_horse : 10 * camel = 24 * horse
axiom horse_ox : 16 * horse = 4 * ox
axiom ox_elephant : 6 * ox = 4 * elephant
axiom elephant_lion : 3 * elephant = 8 * lion
axiom lion_bear : 2 * lion = 6 * bear
axiom bear_cost : 14 * bear = 204000

-- Theorem to prove
theorem camel_cost : camel = 46542.86 := by sorry

end camel_cost_l2135_213553


namespace custom_calculator_results_l2135_213508

-- Define the custom operation *
noncomputable def customOp (a b : ℤ) : ℤ := 2 * a - b

-- Properties of the custom operation
axiom prop_i (a : ℤ) : customOp a a = a
axiom prop_ii (a : ℤ) : customOp a 0 = 2 * a
axiom prop_iii (a b c d : ℤ) : customOp a b + customOp c d = customOp (a + c) (b + d)

-- Theorem to prove
theorem custom_calculator_results :
  (customOp 2 3 + customOp 0 3 = -2) ∧ (customOp 1024 48 = 2000) := by
  sorry

end custom_calculator_results_l2135_213508


namespace max_volume_special_tetrahedron_l2135_213552

/-- A tetrahedron with two vertices on a sphere of radius √10 and two on a concentric sphere of radius 2 -/
structure SpecialTetrahedron where
  /-- The radius of the larger sphere -/
  R : ℝ
  /-- The radius of the smaller sphere -/
  r : ℝ
  /-- Assertion that R = √10 -/
  h_R : R = Real.sqrt 10
  /-- Assertion that r = 2 -/
  h_r : r = 2

/-- The volume of a SpecialTetrahedron -/
def volume (t : SpecialTetrahedron) : ℝ :=
  sorry

/-- The maximum volume of a SpecialTetrahedron is 6√2 -/
theorem max_volume_special_tetrahedron :
  ∀ t : SpecialTetrahedron, volume t ≤ 6 * Real.sqrt 2 :=
sorry

end max_volume_special_tetrahedron_l2135_213552


namespace max_soap_boxes_in_carton_l2135_213561

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Theorem: Maximum number of soap boxes in a carton -/
theorem max_soap_boxes_in_carton (carton soap : BoxDimensions)
    (h_carton : carton = ⟨25, 42, 60⟩)
    (h_soap : soap = ⟨7, 6, 10⟩) :
    (boxVolume carton) / (boxVolume soap) = 150 := by
  sorry


end max_soap_boxes_in_carton_l2135_213561


namespace divisibility_problem_l2135_213525

theorem divisibility_problem :
  {n : ℤ | (n - 2) ∣ (n^2 + 3*n + 27)} = {1, 3, 39, -35} := by
  sorry

end divisibility_problem_l2135_213525


namespace julia_tag_game_l2135_213565

theorem julia_tag_game (tuesday_kids : ℕ) (extra_monday_kids : ℕ) : 
  tuesday_kids = 14 → extra_monday_kids = 8 → tuesday_kids + extra_monday_kids = 22 :=
by
  sorry

end julia_tag_game_l2135_213565


namespace quadratic_coefficient_l2135_213550

theorem quadratic_coefficient (α : ℝ) (p q : ℝ) : 
  (∀ x, x^2 - (α - 2)*x - α - 1 = 0 ↔ x = p ∨ x = q) →
  (∀ a b, a^2 + b^2 ≥ 5 ∧ (a = p ∧ b = q ∨ a = q ∧ b = p) → p^2 + q^2 ≥ 5) →
  p^2 + q^2 = 5 →
  α - 2 = -1 :=
sorry

end quadratic_coefficient_l2135_213550


namespace banquet_food_consumption_l2135_213534

/-- Represents the football banquet scenario -/
structure FootballBanquet where
  /-- The maximum amount of food (in pounds) consumed by any individual guest -/
  max_food_per_guest : ℝ
  /-- The minimum number of guests that attended the banquet -/
  min_guests : ℕ
  /-- The total amount of food (in pounds) consumed at the banquet -/
  total_food_consumed : ℝ

/-- Theorem stating that the total food consumed at the banquet is at least 326 pounds -/
theorem banquet_food_consumption (banquet : FootballBanquet)
  (h1 : banquet.max_food_per_guest ≤ 2)
  (h2 : banquet.min_guests ≥ 163)
  : banquet.total_food_consumed ≥ 326 := by
  sorry

#check banquet_food_consumption

end banquet_food_consumption_l2135_213534


namespace arun_weight_average_l2135_213559

-- Define Arun's weight as a real number
def arun_weight : ℝ := sorry

-- Define the conditions on Arun's weight
def condition1 : Prop := 65 < arun_weight ∧ arun_weight < 72
def condition2 : Prop := 60 < arun_weight ∧ arun_weight < 70
def condition3 : Prop := arun_weight ≤ 68

-- Theorem to prove
theorem arun_weight_average :
  condition1 ∧ condition2 ∧ condition3 →
  (65 + 68) / 2 = 66.5 :=
by sorry

end arun_weight_average_l2135_213559


namespace plywood_cut_perimeter_difference_l2135_213558

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem plywood_cut_perimeter_difference :
  let original : Rectangle := { length := 9, width := 6 }
  let pieces : ℕ := 4
  ∃ (max_piece min_piece : Rectangle),
    (pieces * max_piece.length * max_piece.width = original.length * original.width) ∧
    (pieces * min_piece.length * min_piece.width = original.length * original.width) ∧
    (∀ piece : Rectangle, 
      (pieces * piece.length * piece.width = original.length * original.width) → 
      (perimeter piece ≤ perimeter max_piece ∧ perimeter piece ≥ perimeter min_piece)) ∧
    (perimeter max_piece - perimeter min_piece = 9) := by
  sorry

end plywood_cut_perimeter_difference_l2135_213558


namespace revenue_comparison_l2135_213546

theorem revenue_comparison (base_revenue : ℝ) (projected_increase : ℝ) (actual_decrease : ℝ) :
  projected_increase = 0.2 →
  actual_decrease = 0.25 →
  (base_revenue * (1 - actual_decrease)) / (base_revenue * (1 + projected_increase)) = 0.625 := by
  sorry

end revenue_comparison_l2135_213546


namespace sufficient_not_necessary_condition_l2135_213502

theorem sufficient_not_necessary_condition :
  (∀ a b : ℝ, a > 2 ∧ b > 2 → a + b > 4 ∧ a * b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ a * b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end sufficient_not_necessary_condition_l2135_213502


namespace sum_six_consecutive_integers_l2135_213500

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end sum_six_consecutive_integers_l2135_213500


namespace total_rocks_in_border_l2135_213572

/-- The number of rocks in Mrs. Hilt's garden border -/
def garden_border (placed : ℝ) (additional : ℝ) : ℝ :=
  placed + additional

/-- Theorem stating the total number of rocks in the completed border -/
theorem total_rocks_in_border :
  garden_border 125.0 64.0 = 189.0 := by
  sorry

end total_rocks_in_border_l2135_213572


namespace mold_growth_problem_l2135_213590

/-- Calculates the number of mold spores after a given time period -/
def mold_growth (initial_spores : ℕ) (doubling_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  initial_spores * 2^(elapsed_time / doubling_time)

/-- The mold growth problem -/
theorem mold_growth_problem :
  let initial_spores : ℕ := 50
  let doubling_time : ℕ := 10  -- in minutes
  let elapsed_time : ℕ := 70   -- time from 9:00 a.m. to 10:10 a.m. in minutes
  mold_growth initial_spores doubling_time elapsed_time = 6400 :=
by
  sorry

end mold_growth_problem_l2135_213590


namespace count_valid_numbers_l2135_213566

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧
  n > 80 ∧
  is_prime (n / 10) ∧
  is_even (n % 10)

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_number n) ∧ S.card = 5 :=
sorry

end count_valid_numbers_l2135_213566


namespace two_fifths_in_nine_thirds_l2135_213510

theorem two_fifths_in_nine_thirds : (9 / 3) / (2 / 5) = 15 / 2 := by
  sorry

end two_fifths_in_nine_thirds_l2135_213510


namespace charity_ticket_sales_l2135_213595

theorem charity_ticket_sales (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 140)
  (h_total_revenue : total_revenue = 2001) :
  ∃ (full_price : ℕ) (half_price : ℕ) (full_price_tickets : ℕ) (half_price_tickets : ℕ),
    full_price > 0 ∧
    half_price = full_price / 2 ∧
    full_price_tickets + half_price_tickets = total_tickets ∧
    full_price_tickets * full_price + half_price_tickets * half_price = total_revenue ∧
    full_price_tickets * full_price = 782 :=
by sorry

end charity_ticket_sales_l2135_213595


namespace benjamin_walks_95_miles_l2135_213541

/-- Represents the total miles Benjamin walks in a week -/
def total_miles_walked : ℕ :=
  let work_distance := 6
  let dog_walk_distance := 2
  let friend_distance := 1
  let store_distance := 3
  let work_days := 5
  let dog_walks_per_day := 2
  let days_in_week := 7
  let store_visits := 2
  let friend_visits := 1

  (2 * work_distance * work_days) + 
  (dog_walk_distance * dog_walks_per_day * days_in_week) + 
  (2 * store_distance * store_visits) + 
  (2 * friend_distance * friend_visits)

theorem benjamin_walks_95_miles : total_miles_walked = 95 := by
  sorry

end benjamin_walks_95_miles_l2135_213541


namespace evaluate_polynomial_at_negative_two_l2135_213554

theorem evaluate_polynomial_at_negative_two :
  let y : ℤ := -2
  y^3 - y^2 + 2*y + 4 = -12 := by
sorry

end evaluate_polynomial_at_negative_two_l2135_213554


namespace conditional_probability_rain_wind_l2135_213528

theorem conditional_probability_rain_wind (P_rain P_wind_and_rain : ℚ) 
  (h1 : P_rain = 4 / 15)
  (h2 : P_wind_and_rain = 1 / 10) :
  P_wind_and_rain / P_rain = 3 / 8 := by
  sorry

end conditional_probability_rain_wind_l2135_213528


namespace sin_product_equals_one_sixteenth_l2135_213556

theorem sin_product_equals_one_sixteenth : 
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * 
  Real.sin (54 * π / 180) * Real.sin (78 * π / 180) = 1/16 := by
  sorry

end sin_product_equals_one_sixteenth_l2135_213556


namespace sin_theta_value_l2135_213529

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 5 * Real.cos θ) 
  (h2 : 0 < θ) 
  (h3 : θ < Real.pi / 2) : 
  Real.sin θ = 1 := by sorry

end sin_theta_value_l2135_213529


namespace three_fraction_equality_l2135_213543

theorem three_fraction_equality (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hdiff : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (heq : (y + 1) / (x - z + 1) = (x + y + 2) / (z + 2) ∧ 
         (x + y + 2) / (z + 2) = (x + 1) / (y + 1)) : 
  (x + 1) / (y + 1) = 2 := by
  sorry

end three_fraction_equality_l2135_213543


namespace f_properties_l2135_213511

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / (x + 1)

def tangent_slope (a : ℝ) (x : ℝ) : ℝ :=
  1 / x - a / ((x + 1) ^ 2)

def critical_point (a : ℝ) : ℝ :=
  (a - 2 + Real.sqrt ((a - 2) ^ 2 + 4)) / 2

theorem f_properties (a : ℝ) (h : a ≥ 0) :
  (tangent_slope 3 1 = 1/4) ∧
  (∀ x > 0, f a x ≤ (2016 - a) * x^3 + (x^2 + a - 1) / (x + 1) →
    (∃ x > 0, (tangent_slope a x = 0) → 4 < a ∧ a ≤ 2016)) :=
sorry

end

end f_properties_l2135_213511


namespace largest_number_with_digit_sum_20_l2135_213503

def digit_sum (n : Nat) : Nat :=
  Nat.digits 10 n |>.sum

def all_digits_different (n : Nat) : Prop :=
  (Nat.digits 10 n).Nodup

def no_zero_digit (n : Nat) : Prop :=
  0 ∉ Nat.digits 10 n

theorem largest_number_with_digit_sum_20 :
  ∀ n : Nat,
    (digit_sum n = 20 ∧
     all_digits_different n ∧
     no_zero_digit n) →
    n ≤ 9821 :=
by sorry

end largest_number_with_digit_sum_20_l2135_213503


namespace sides_when_k_is_two_k_values_l2135_213517

/-- Represents a regular pyramid -/
structure RegularPyramid where
  n : ℕ  -- number of sides of the base
  α : ℝ  -- dihedral angle at the base
  β : ℝ  -- angle formed by lateral edges with the base plane
  k : ℝ  -- relationship constant between α and β
  h1 : α > 0
  h2 : β > 0
  h3 : k > 0
  h4 : n ≥ 3
  h5 : Real.tan α = k * Real.tan β
  h6 : k = 1 / Real.cos (π / n)

/-- The number of sides of the base is 3 when k = 2 -/
theorem sides_when_k_is_two (p : RegularPyramid) : p.k = 2 → p.n = 3 := by sorry

/-- The possible values of k are given by 1 / cos(π/n) where n ≥ 3 -/
theorem k_values (p : RegularPyramid) : 
  ∃ (n : ℕ), n ≥ 3 ∧ p.k = 1 / Real.cos (π / n) := by sorry

end sides_when_k_is_two_k_values_l2135_213517


namespace quadratic_ratio_l2135_213545

/-- Given a quadratic polynomial x^2 + 1500x + 1800, prove that when written in the form (x+a)^2 + d,
    the ratio d/a equals -560700/750. -/
theorem quadratic_ratio (x : ℝ) :
  ∃ (a d : ℝ), x^2 + 1500*x + 1800 = (x + a)^2 + d ∧ d / a = -560700 / 750 :=
by sorry

end quadratic_ratio_l2135_213545


namespace birth_year_proof_l2135_213513

/-- A person born in the first half of the 19th century whose age was x in the year x^2 was born in 1806 -/
theorem birth_year_proof (x : ℕ) (h1 : 1800 < x^2) (h2 : x^2 < 1850) (h3 : x^2 - x = 1806) : 
  x^2 - x = 1806 := by sorry

end birth_year_proof_l2135_213513


namespace abs_x_less_than_2_sufficient_not_necessary_l2135_213524

theorem abs_x_less_than_2_sufficient_not_necessary :
  (∀ x : ℝ, (|x| < 2 ↔ -2 < x ∧ x < 2)) →
  (∀ x : ℝ, (x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3)) →
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  ¬(∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) :=
by sorry

end abs_x_less_than_2_sufficient_not_necessary_l2135_213524


namespace opposite_of_negative_negative_five_l2135_213537

theorem opposite_of_negative_negative_five :
  -(-(5 : ℤ)) = -5 := by sorry

end opposite_of_negative_negative_five_l2135_213537


namespace soccer_team_starters_l2135_213520

theorem soccer_team_starters (n : ℕ) (k : ℕ) (q : ℕ) (m : ℕ) 
  (h1 : n = 16) 
  (h2 : k = 7) 
  (h3 : q = 4) 
  (h4 : m = 1) :
  (q.choose m) * ((n - q).choose (k - m)) = 3696 := by
  sorry

end soccer_team_starters_l2135_213520


namespace remainder_plus_fraction_equals_result_l2135_213596

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem remainder_plus_fraction_equals_result :
  rem (5/7) (-3/4) + 1/14 = 1/28 := by
  sorry

end remainder_plus_fraction_equals_result_l2135_213596


namespace inverse_mod_53_l2135_213598

theorem inverse_mod_53 (h : (19⁻¹ : ZMod 53) = 31) : (44⁻¹ : ZMod 53) = 22 := by
  sorry

end inverse_mod_53_l2135_213598


namespace block_distribution_l2135_213557

theorem block_distribution (n : ℕ) (h : n > 0) (h_divides : n ∣ 49) :
  ∃ (blocks_per_color : ℕ), blocks_per_color > 0 ∧ blocks_per_color * n = 49 := by
  sorry

end block_distribution_l2135_213557


namespace exactly_three_props_true_l2135_213526

/-- Property P for a sequence -/
def has_property_P (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i < j → j ≤ n → (∃ k ≤ n, a k = a j + a i) ∨ (∃ k ≤ n, a k = a j - a i)

/-- The sequence is strictly increasing and starts with a non-negative number -/
def is_valid_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) ∧ 0 ≤ a 1

/-- Proposition 1: The sequence 0, 2, 4, 6 has property P -/
def prop_1 : Prop :=
  let a : ℕ → ℕ := fun i => 2 * (i - 1)
  has_property_P a 4

/-- Proposition 2: If sequence A has property P, then a₁ = 0 -/
def prop_2 : Prop :=
  ∀ a : ℕ → ℕ, ∀ n ≥ 3, is_valid_sequence a n → has_property_P a n → a 1 = 0

/-- Proposition 3: If sequence A has property P and a₁ ≠ 0, then aₙ - aₙ₋ₖ = aₖ for k = 1, 2, ..., n-1 -/
def prop_3 : Prop :=
  ∀ a : ℕ → ℕ, ∀ n ≥ 3, is_valid_sequence a n → has_property_P a n → a 1 ≠ 0 →
    ∀ k, 1 ≤ k → k < n → a n - a (n - k) = a k

/-- Proposition 4: If the sequence a₁, a₂, a₃ (0 ≤ a₁ < a₂ < a₃) has property P, then a₃ = a₁ + a₂ -/
def prop_4 : Prop :=
  ∀ a : ℕ → ℕ, is_valid_sequence a 3 → has_property_P a 3 → a 3 = a 1 + a 2

theorem exactly_three_props_true : (prop_1 ∧ ¬prop_2 ∧ prop_3 ∧ prop_4) := by sorry

end exactly_three_props_true_l2135_213526


namespace definite_integral_abs_quadratic_l2135_213592

theorem definite_integral_abs_quadratic : ∫ x in (-2)..2, |x^2 - 2*x| = 8 := by
  sorry

end definite_integral_abs_quadratic_l2135_213592


namespace percentage_problem_l2135_213501

theorem percentage_problem (p : ℝ) (h1 : 0.25 * 820 = (p/100) * 1500 - 20) : p = 15 := by
  sorry

end percentage_problem_l2135_213501


namespace doll_difference_proof_l2135_213539

-- Define the number of dolls for each person
def geraldine_dolls : ℝ := 2186.0
def jazmin_dolls : ℝ := 1209.0

-- Define the difference in dolls
def doll_difference : ℝ := geraldine_dolls - jazmin_dolls

-- Theorem statement
theorem doll_difference_proof : doll_difference = 977.0 := by
  sorry

end doll_difference_proof_l2135_213539


namespace y_intercept_of_line_l2135_213530

def line (x : ℝ) : ℝ := x - 2

theorem y_intercept_of_line :
  line 0 = -2 := by sorry

end y_intercept_of_line_l2135_213530


namespace statement_c_is_false_l2135_213577

theorem statement_c_is_false : ¬(∀ (p q : Prop), ¬(p ∧ q) → (¬p ∧ ¬q)) := by
  sorry

end statement_c_is_false_l2135_213577


namespace intersection_theorem_l2135_213597

-- Define set A
def A : Set ℝ := {x | (x + 2) / (x - 2) ≤ 0}

-- Define set B
def B : Set ℝ := {x | |x - 1| < 2}

-- Define the complement of A with respect to ℝ
def not_A : Set ℝ := {x | x ∉ A}

-- Define the intersection of B and the complement of A
def B_intersect_not_A : Set ℝ := B ∩ not_A

-- Theorem statement
theorem intersection_theorem : B_intersect_not_A = {x | 2 ≤ x ∧ x < 3} := by sorry

end intersection_theorem_l2135_213597


namespace inequality_problem_l2135_213514

theorem inequality_problem (a b x y : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_sum : a + b = 1) : 
  (a*x + b*y) * (b*x + a*y) ≥ x*y := by
sorry

end inequality_problem_l2135_213514


namespace lily_painting_rate_l2135_213594

/-- Represents the number of cups Gina can paint per hour -/
structure PaintingRate where
  roses : ℕ
  lilies : ℕ

/-- Represents an order of cups -/
structure Order where
  roses : ℕ
  lilies : ℕ

theorem lily_painting_rate 
  (gina_rate : PaintingRate)
  (order : Order)
  (total_payment : ℕ)
  (hourly_rate : ℕ)
  (h1 : gina_rate.roses = 6)
  (h2 : order.roses = 6)
  (h3 : order.lilies = 14)
  (h4 : total_payment = 90)
  (h5 : hourly_rate = 30) :
  gina_rate.lilies = 7 := by
  sorry

end lily_painting_rate_l2135_213594


namespace isosceles_triangle_side_lengths_l2135_213533

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of the base or the equal side
  h : a > 0 ∧ b > 0  -- Lengths are positive

/-- Checks if the given side lengths form a valid triangle -/
def is_valid_triangle (t : IsoscelesTriangle) : Prop :=
  t.a + t.b > t.a ∧ t.a + t.a > t.b

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.a + t.b

theorem isosceles_triangle_side_lengths :
  ∀ t : IsoscelesTriangle,
    is_valid_triangle t →
    perimeter t = 17 →
    (t.a = 4 ∨ t.b = 4) →
    ((t.a = 6 ∧ t.b = 5) ∨ (t.a = 5 ∧ t.b = 7)) :=
by sorry

end isosceles_triangle_side_lengths_l2135_213533


namespace percentage_same_grade_is_42_5_l2135_213506

/-- The total number of students in the class -/
def total_students : ℕ := 40

/-- The number of students who received an 'A' on both tests -/
def same_grade_A : ℕ := 3

/-- The number of students who received a 'B' on both tests -/
def same_grade_B : ℕ := 5

/-- The number of students who received a 'C' on both tests -/
def same_grade_C : ℕ := 6

/-- The number of students who received a 'D' on both tests -/
def same_grade_D : ℕ := 2

/-- The number of students who received an 'E' on both tests -/
def same_grade_E : ℕ := 1

/-- The total number of students who received the same grade on both tests -/
def total_same_grade : ℕ := same_grade_A + same_grade_B + same_grade_C + same_grade_D + same_grade_E

/-- The percentage of students who received the same grade on both tests -/
def percentage_same_grade : ℚ := (total_same_grade : ℚ) / (total_students : ℚ) * 100

theorem percentage_same_grade_is_42_5 : percentage_same_grade = 42.5 := by
  sorry

end percentage_same_grade_is_42_5_l2135_213506


namespace trapezoid_area_sum_l2135_213562

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the sum of all possible areas of a trapezoid -/
def sum_of_areas (t : Trapezoid) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_square_prime (n : ℕ) : Prop := sorry

/-- Main theorem statement -/
theorem trapezoid_area_sum (t : Trapezoid) :
  t.side1 = 4 ∧ t.side2 = 6 ∧ t.side3 = 8 ∧ t.side4 = 10 →
  ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
    sum_of_areas t = r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃ ∧
    not_divisible_by_square_prime n₁ ∧
    not_divisible_by_square_prime n₂ ∧
    ⌊r₁ + r₂ + r₃ + n₁ + n₂⌋ = 18 := by
  sorry

end trapezoid_area_sum_l2135_213562


namespace ball_drawing_theorem_l2135_213599

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  white : Nat
  red : Nat
  black : Nat

/-- The minimum number of balls to draw to ensure at least one of each color -/
def minDrawForAllColors (counts : BallCounts) : Nat :=
  counts.white + counts.red + counts.black - 2

/-- The minimum number of balls to draw to ensure 10 balls of one color -/
def minDrawForTenOfOneColor (counts : BallCounts) : Nat :=
  min counts.white counts.red + min counts.white counts.black + 
  min counts.red counts.black + 10 - 1

/-- Theorem stating the correct answers for the given ball counts -/
theorem ball_drawing_theorem (counts : BallCounts) 
  (h1 : counts.white = 5) (h2 : counts.red = 12) (h3 : counts.black = 20) : 
  minDrawForAllColors counts = 33 ∧ minDrawForTenOfOneColor counts = 24 := by
  sorry

#eval minDrawForAllColors ⟨5, 12, 20⟩
#eval minDrawForTenOfOneColor ⟨5, 12, 20⟩

end ball_drawing_theorem_l2135_213599


namespace problem_statement_l2135_213567

theorem problem_statement (x y : ℝ) : 
  (x - 1)^2 + |y + 1| = 0 → 2*(x^2 - y^2 + 1) - 2*(x^2 + y^2) + x*y = -3 := by
sorry

end problem_statement_l2135_213567


namespace hyperbola_theorem_l2135_213576

/-- A hyperbola C that shares a common asymptote with x^2 - 2y^2 = 2 and passes through (2, -2) -/
structure Hyperbola where
  -- The equation of the hyperbola in the form y^2/a^2 - x^2/b^2 = 1
  a : ℝ
  b : ℝ
  -- The hyperbola passes through (2, -2)
  point_condition : (2 : ℝ)^2 / b^2 - (-2 : ℝ)^2 / a^2 = 1
  -- The hyperbola shares a common asymptote with x^2 - 2y^2 = 2
  asymptote_condition : a^2 / b^2 = 2

/-- Properties of the hyperbola C -/
def hyperbola_properties (C : Hyperbola) : Prop :=
  -- The equation of C is y^2/2 - x^2/4 = 1
  C.a^2 = 2 ∧ C.b^2 = 4 ∧
  -- The eccentricity of C is √3
  Real.sqrt ((C.a^2 + C.b^2) / C.a^2) = Real.sqrt 3 ∧
  -- The asymptotes of C are y = ±(√2/2)x
  ∀ (x y : ℝ), (y = Real.sqrt 2 / 2 * x ∨ y = -Real.sqrt 2 / 2 * x) ↔ 
    (y^2 / C.a^2 - x^2 / C.b^2 = 0)

/-- Main theorem: The hyperbola C satisfies the required properties -/
theorem hyperbola_theorem (C : Hyperbola) : hyperbola_properties C :=
sorry

end hyperbola_theorem_l2135_213576


namespace curve_equation_proof_l2135_213516

theorem curve_equation_proof :
  let x : ℝ → ℝ := λ t => 3 * Real.cos t - 2 * Real.sin t
  let y : ℝ → ℝ := λ t => 3 * Real.sin t
  let a : ℝ := 1 / 9
  let b : ℝ := -4 / 27
  let c : ℝ := 5 / 81
  let d : ℝ := 0
  let e : ℝ := 1 / 3
  ∀ t : ℝ, a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 + d * (x t) + e * (y t) = 1 :=
by sorry

end curve_equation_proof_l2135_213516


namespace range_of_a_l2135_213527

-- Define the sets S and T
def S : Set ℝ := {x | |x - 1| + |x + 2| > 5}
def T (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}

-- State the theorem
theorem range_of_a (a : ℝ) : S ∪ T a = Set.univ → -2 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l2135_213527


namespace lowest_score_problem_l2135_213591

theorem lowest_score_problem (scores : Finset ℕ) (highest lowest : ℕ) :
  Finset.card scores = 15 →
  highest ∈ scores →
  lowest ∈ scores →
  highest = 100 →
  (Finset.sum scores id) / 15 = 85 →
  ((Finset.sum scores id) - highest - lowest) / 13 = 86 →
  lowest = 57 := by
  sorry

end lowest_score_problem_l2135_213591


namespace davids_physics_marks_l2135_213551

/-- Given David's marks in various subjects and his average, prove his marks in Physics --/
theorem davids_physics_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (total_subjects : ℕ)
  (h1 : english_marks = 86)
  (h2 : math_marks = 85)
  (h3 : chemistry_marks = 87)
  (h4 : biology_marks = 85)
  (h5 : average_marks = 85)
  (h6 : total_subjects = 5)
  : ∃ (physics_marks : ℕ),
    physics_marks = average_marks * total_subjects - (english_marks + math_marks + chemistry_marks + biology_marks) ∧
    physics_marks = 82 :=
by sorry

end davids_physics_marks_l2135_213551


namespace sequence_sum_comparison_l2135_213521

theorem sequence_sum_comparison (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ k, k > 0 → S k = -a k - (1/2)^(k-1) + 2) : 
  (n ≥ 5 → S n > 2 - 1/(n-1)) ∧ 
  ((n = 3 ∨ n = 4) → S n < 2 - 1/(n-1)) := by
  sorry

end sequence_sum_comparison_l2135_213521


namespace polynomial_division_remainder_l2135_213574

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  x^30 + x^24 + x^18 + x^12 + x^6 + 1 = (x^4 + x^3 + x^2 + x + 1) * q + 1 := by
  sorry

end polynomial_division_remainder_l2135_213574


namespace complex_fraction_equals_i_l2135_213588

theorem complex_fraction_equals_i : (1 + Complex.I) / (1 - Complex.I) = Complex.I := by sorry

end complex_fraction_equals_i_l2135_213588


namespace line_slope_l2135_213547

theorem line_slope (x y : ℝ) (h : x + 2 * y - 3 = 0) : 
  ∃ m b : ℝ, m = -1/2 ∧ y = m * x + b :=
sorry

end line_slope_l2135_213547


namespace initial_pens_l2135_213589

theorem initial_pens (initial : ℕ) (mike_gives : ℕ) (cindy_doubles : ℕ → ℕ) (sharon_takes : ℕ) (final : ℕ) : 
  mike_gives = 22 →
  cindy_doubles = (· * 2) →
  sharon_takes = 19 →
  final = 39 →
  cindy_doubles (initial + mike_gives) - sharon_takes = final →
  initial = 7 := by
sorry

end initial_pens_l2135_213589


namespace distribution_law_l2135_213549

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  h_x_lt : x₁ < x₂
  h_p_bound : 0 ≤ p₁ ∧ p₁ ≤ 1

/-- Expectation of a DiscreteRV -/
def expectation (X : DiscreteRV) : ℝ := X.x₁ * X.p₁ + X.x₂ * (1 - X.p₁)

/-- Variance of a DiscreteRV -/
def variance (X : DiscreteRV) : ℝ :=
  X.p₁ * (X.x₁ - expectation X)^2 + (1 - X.p₁) * (X.x₂ - expectation X)^2

/-- Theorem stating the distribution law of the given discrete random variable -/
theorem distribution_law (X : DiscreteRV)
  (h_p₁ : X.p₁ = 0.5)
  (h_expectation : expectation X = 3.5)
  (h_variance : variance X = 0.25) :
  X.x₁ = 3 ∧ X.x₂ = 4 :=
sorry

end distribution_law_l2135_213549


namespace distribute_six_balls_three_boxes_l2135_213548

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end distribute_six_balls_three_boxes_l2135_213548


namespace ten_parabolas_regions_l2135_213523

/-- The number of regions a circle can be divided into by n parabolas -/
def circle_regions (n : ℕ) : ℕ := 2 * n^2 + 1

/-- Theorem stating that 10 parabolas divide a circle into 201 regions -/
theorem ten_parabolas_regions : circle_regions 10 = 201 := by
  sorry

end ten_parabolas_regions_l2135_213523


namespace carls_garden_area_l2135_213522

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  total_posts : Nat
  post_spacing : Nat
  longer_side_posts : Nat
  shorter_side_posts : Nat

/-- Calculates the area of the garden given the specifications -/
def calculate_area (g : Garden) : Nat :=
  (g.shorter_side_posts - 1) * g.post_spacing * 
  (g.longer_side_posts - 1) * g.post_spacing

/-- Theorem stating the area of Carl's garden -/
theorem carls_garden_area : 
  ∀ g : Garden, 
    g.total_posts = 24 ∧ 
    g.post_spacing = 5 ∧ 
    g.longer_side_posts = 2 * g.shorter_side_posts ∧
    g.longer_side_posts + g.shorter_side_posts = g.total_posts + 2 →
    calculate_area g = 900 := by
  sorry

end carls_garden_area_l2135_213522


namespace sum_10_terms_l2135_213515

/-- An arithmetic sequence with a₂ = 3 and a₉ = 17 -/
def arithmetic_seq (n : ℕ) : ℝ :=
  sorry

/-- The sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ :=
  sorry

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence is 100 -/
theorem sum_10_terms : S 10 = 100 :=
  sorry

end sum_10_terms_l2135_213515
