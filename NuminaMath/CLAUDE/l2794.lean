import Mathlib

namespace NUMINAMATH_CALUDE_max_value_implies_a_l2794_279494

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - 3

theorem max_value_implies_a (a : ℝ) (h_a : a ≠ 0) :
  (∀ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x ≤ 1) ∧
  (∃ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x = 1) →
  a = 3/4 ∨ a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l2794_279494


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2794_279462

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = a n * r ∧ a n > 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2794_279462


namespace NUMINAMATH_CALUDE_rectangle_width_l2794_279415

/-- Given a rectangle where the length is 3 times the width and the area is 108 square inches,
    prove that the width is 6 inches. -/
theorem rectangle_width (w : ℝ) (h1 : w > 0) (h2 : 3 * w * w = 108) : w = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l2794_279415


namespace NUMINAMATH_CALUDE_excess_donation_l2794_279448

/-- Trader's profit calculation -/
def trader_profit : ℝ := 1200

/-- Allocation percentage for next shipment -/
def allocation_percentage : ℝ := 0.60

/-- Family donation amount -/
def family_donation : ℝ := 250

/-- Friends donation calculation -/
def friends_donation : ℝ := family_donation * 1.20

/-- Local association donation calculation -/
def local_association_donation : ℝ := (family_donation + friends_donation) * 1.5

/-- Total donations received -/
def total_donations : ℝ := family_donation + friends_donation + local_association_donation

/-- Allocated amount for next shipment -/
def allocated_amount : ℝ := trader_profit * allocation_percentage

/-- Theorem: The difference between total donations and allocated amount is $655 -/
theorem excess_donation : total_donations - allocated_amount = 655 := by sorry

end NUMINAMATH_CALUDE_excess_donation_l2794_279448


namespace NUMINAMATH_CALUDE_number_thought_of_l2794_279488

theorem number_thought_of (x : ℝ) : (6 * x^2 - 10) / 3 + 15 = 95 → x = 5 * Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l2794_279488


namespace NUMINAMATH_CALUDE_unique_solution_for_x_l2794_279476

theorem unique_solution_for_x (x y z : ℤ) 
  (h1 : x > y ∧ y > z ∧ z > 0)
  (h2 : x + y + z + x*y + y*z + z*x = 104) : 
  x = 6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_x_l2794_279476


namespace NUMINAMATH_CALUDE_production_days_l2794_279438

theorem production_days (n : ℕ) 
  (h1 : (50 : ℝ) * n = n * 50)
  (h2 : (50 : ℝ) * n + 115 = (n + 1) * 55) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l2794_279438


namespace NUMINAMATH_CALUDE_min_value_theorem_l2794_279404

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x + 2)^2 / (y - 2) + (y + 2)^2 / (x - 2) ≥ 50 ∧
  ((x + 2)^2 / (y - 2) + (y + 2)^2 / (x - 2) = 50 ↔ x = 3 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2794_279404


namespace NUMINAMATH_CALUDE_chris_age_l2794_279428

/-- Represents the ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℝ
  ben : ℝ
  chris : ℝ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 12
  (ages.amy + ages.ben + ages.chris) / 3 = 12 ∧
  -- Five years ago, Chris was the same age as Amy is now
  ages.chris - 5 = ages.amy ∧
  -- In 4 years, Ben's age will be 3/4 of Amy's age at that time
  ages.ben + 4 = (3/4) * (ages.amy + 4)

/-- The theorem to be proved -/
theorem chris_age (ages : Ages) :
  problem_conditions ages → ages.chris = 15.55 := by
  sorry

end NUMINAMATH_CALUDE_chris_age_l2794_279428


namespace NUMINAMATH_CALUDE_exists_homogeneous_polynomial_for_irreducible_lattice_points_l2794_279461

-- Define an irreducible lattice point
def irreducible_lattice_point (p : ℤ × ℤ) : Prop :=
  Int.gcd p.1 p.2 = 1

-- Define a homogeneous polynomial with integer coefficients
def homogeneous_polynomial (f : ℤ → ℤ → ℤ) (d : ℕ) : Prop :=
  ∀ (c : ℤ) (x y : ℤ), f (c * x) (c * y) = c^d * f x y

-- The main theorem
theorem exists_homogeneous_polynomial_for_irreducible_lattice_points 
  (S : Finset (ℤ × ℤ)) (h : ∀ p ∈ S, irreducible_lattice_point p) :
  ∃ (f : ℤ → ℤ → ℤ) (d : ℕ), 
    d ≥ 1 ∧ 
    homogeneous_polynomial f d ∧ 
    (∀ p ∈ S, f p.1 p.2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_exists_homogeneous_polynomial_for_irreducible_lattice_points_l2794_279461


namespace NUMINAMATH_CALUDE_polynomial_constant_term_product_l2794_279414

variable (p q r : ℝ[X])

theorem polynomial_constant_term_product 
  (h1 : r = p * q)
  (h2 : p.coeff 0 = 6)
  (h3 : r.coeff 0 = -18) :
  q.eval 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_constant_term_product_l2794_279414


namespace NUMINAMATH_CALUDE_circle_tangent_to_ellipse_l2794_279499

/-- Two circles of radius s are externally tangent to each other and internally tangent to the ellipse x^2 + 4y^2 = 8. The radius s of the circles is √(3/2). -/
theorem circle_tangent_to_ellipse (s : ℝ) : 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 8 ∧ (x - s)^2 + y^2 = s^2) → s = Real.sqrt (3/2) := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_ellipse_l2794_279499


namespace NUMINAMATH_CALUDE_french_exam_min_words_to_learn_l2794_279492

/-- The minimum number of words to learn for a 90% score on a French vocabulary exam -/
theorem french_exam_min_words_to_learn :
  ∀ (total_words : ℕ) (guess_success_rate : ℚ) (target_score : ℚ),
    total_words = 800 →
    guess_success_rate = 1/10 →
    target_score = 9/10 →
    ∃ (words_to_learn : ℕ),
      words_to_learn ≥ 712 ∧
      (words_to_learn : ℚ) / total_words +
        guess_success_rate * ((total_words : ℚ) - words_to_learn) / total_words ≥ target_score :=
by sorry

end NUMINAMATH_CALUDE_french_exam_min_words_to_learn_l2794_279492


namespace NUMINAMATH_CALUDE_envelope_width_l2794_279481

/-- Given a rectangular envelope with an area of 36 square inches and a height of 6 inches,
    prove that its width is 6 inches. -/
theorem envelope_width (area : ℝ) (height : ℝ) (width : ℝ) 
    (h1 : area = 36) 
    (h2 : height = 6) 
    (h3 : area = width * height) : 
  width = 6 := by
  sorry

end NUMINAMATH_CALUDE_envelope_width_l2794_279481


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l2794_279435

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- State the theorem
theorem opposite_of_negative_six :
  opposite (-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l2794_279435


namespace NUMINAMATH_CALUDE_triangle_area_l2794_279451

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) - 2 * Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sqrt 3

theorem triangle_area (ω : ℝ) (A B C : ℝ) (a b c : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) →
  0 < C ∧ C < π / 2 →
  f 1 C = Real.sqrt 3 →
  c = 3 →
  Real.sin B = 2 * Real.sin A →
  (1 / 2 : ℝ) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2794_279451


namespace NUMINAMATH_CALUDE_sum_of_positive_factors_36_l2794_279484

-- Define the sum of positive factors function
def sumOfPositiveFactors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_positive_factors_36 : sumOfPositiveFactors 36 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_positive_factors_36_l2794_279484


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2794_279402

theorem sum_of_squares_of_roots (a b c : ℚ) (h1 : a = 2) (h2 : b = 5) (h3 : c = -12) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = 73/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2794_279402


namespace NUMINAMATH_CALUDE_point_on_k_graph_l2794_279444

-- Define the functions f and k
variable (f : ℝ → ℝ)
variable (k : ℝ → ℝ)

-- State the theorem
theorem point_on_k_graph (h1 : f 4 = 8) (h2 : ∀ x, k x = (f x)^3) :
  ∃ x y : ℝ, k x = y ∧ x + y = 516 := by
sorry

end NUMINAMATH_CALUDE_point_on_k_graph_l2794_279444


namespace NUMINAMATH_CALUDE_xyz_sum_l2794_279489

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 24) (hxz : x * z = 48) (hyz : y * z = 72) :
  x + y + z = 22 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_l2794_279489


namespace NUMINAMATH_CALUDE_smallest_positive_integer_d_l2794_279412

theorem smallest_positive_integer_d : ∃ d : ℕ+, d = 4 ∧
  (∀ d' : ℕ+, d' < d →
    ¬∃ x y : ℝ, x^2 + y^2 = 100 ∧ y = 2*x + d' ∧ x^2 + y^2 = 100 * d') ∧
  ∃ x y : ℝ, x^2 + y^2 = 100 ∧ y = 2*x + d ∧ x^2 + y^2 = 100 * d :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_d_l2794_279412


namespace NUMINAMATH_CALUDE_mode_and_median_of_data_l2794_279405

def data : List ℕ := [6, 8, 3, 6, 4, 6, 5]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem mode_and_median_of_data :
  mode data = 6 ∧ median data = 6 := by sorry

end NUMINAMATH_CALUDE_mode_and_median_of_data_l2794_279405


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l2794_279437

theorem buckingham_palace_visitors 
  (total_visitors : ℕ) 
  (previous_day_visitors : ℕ) 
  (today_visitors : ℕ) 
  (h1 : total_visitors = 949) 
  (h2 : previous_day_visitors = 703) 
  (h3 : today_visitors > 0) 
  (h4 : total_visitors = previous_day_visitors + today_visitors) : 
  today_visitors = 246 := by
sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l2794_279437


namespace NUMINAMATH_CALUDE_point_on_x_axis_l2794_279470

/-- Given a point P with coordinates (4, 2a+10), prove that if P lies on the x-axis, then a = -5 -/
theorem point_on_x_axis (a : ℝ) : 
  let P : ℝ × ℝ := (4, 2*a + 10)
  (P.2 = 0) → a = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l2794_279470


namespace NUMINAMATH_CALUDE_function_behavior_l2794_279441

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 10

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 2*x - 6

-- Theorem statement
theorem function_behavior :
  ∃ c ∈ Set.Ioo 2 4, 
    (∀ x ∈ Set.Ioo 2 c, (f' x < 0)) ∧ 
    (∀ x ∈ Set.Ioo c 4, (f' x > 0)) :=
sorry


end NUMINAMATH_CALUDE_function_behavior_l2794_279441


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2794_279473

theorem sqrt_equation_solution (y : ℝ) : 
  (Real.sqrt 1.21) / (Real.sqrt y) + (Real.sqrt 1.44) / (Real.sqrt 0.49) = 2.9365079365079367 → 
  y = 0.81 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2794_279473


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l2794_279417

/-- 
Given a sum of money that doubles itself in 10 years at simple interest,
prove that the rate percent per annum is 10%.
-/
theorem simple_interest_rate_for_doubling (P : ℝ) (h : P > 0) : 
  ∃ (R : ℝ), R > 0 ∧ R ≤ 100 ∧ P + (P * R * 10) / 100 = 2 * P ∧ R = 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l2794_279417


namespace NUMINAMATH_CALUDE_three_heads_probability_l2794_279480

/-- A fair coin has a probability of 1/2 for heads on a single flip -/
def fair_coin_prob : ℚ := 1/2

/-- The probability of getting three heads in three flips of a fair coin -/
def three_heads_prob : ℚ := fair_coin_prob * fair_coin_prob * fair_coin_prob

/-- Theorem: The probability of getting three heads in three flips of a fair coin is 1/8 -/
theorem three_heads_probability : three_heads_prob = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_probability_l2794_279480


namespace NUMINAMATH_CALUDE_salary_calculation_l2794_279478

/-- Given a series of salary changes and a final salary, calculate the original salary --/
theorem salary_calculation (S : ℝ) : 
  S * 1.12 * 0.93 * 1.15 * 0.90 = 5204.21 → S = 5504.00 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l2794_279478


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2794_279439

theorem sin_45_degrees :
  let r : ℝ := 1  -- radius of the unit circle
  let θ : ℝ := Real.pi / 4  -- 45° in radians
  let Q : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)  -- point on the circle at 45°
  let E : ℝ × ℝ := (Q.1, 0)  -- foot of the perpendicular from Q to x-axis
  Real.sin θ = 1 / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2794_279439


namespace NUMINAMATH_CALUDE_switching_strategy_wins_more_than_half_l2794_279424

structure ThreeBoxGame where
  boxes : Fin 3 → Bool  -- True if box contains prize, False if empty
  prize_exists : ∃ i, boxes i = true
  two_empty : ∃ i j, i ≠ j ∧ boxes i = false ∧ boxes j = false

def initial_choice (game : ThreeBoxGame) : Fin 3 :=
  sorry

def host_opens (game : ThreeBoxGame) (choice : Fin 3) : Fin 3 :=
  sorry

def switch (initial : Fin 3) (opened : Fin 3) : Fin 3 :=
  sorry

def probability_of_winning_by_switching (game : ThreeBoxGame) : ℝ :=
  sorry

theorem switching_strategy_wins_more_than_half :
  ∀ game : ThreeBoxGame, probability_of_winning_by_switching game > 1/2 :=
sorry

end NUMINAMATH_CALUDE_switching_strategy_wins_more_than_half_l2794_279424


namespace NUMINAMATH_CALUDE_polynomial_value_l2794_279410

theorem polynomial_value (x : ℝ) (h : x = (1 + Real.sqrt 1994) / 2) :
  (4 * x^3 - 1997 * x - 1994)^20001 = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l2794_279410


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2794_279460

/-- The area of the triangle formed by y = 3x - 6, y = -4x + 24, and the y-axis -/
theorem triangle_area : ℝ → Prop :=
  λ area : ℝ =>
    let line1 : ℝ → ℝ := λ x => 3 * x - 6
    let line2 : ℝ → ℝ := λ x => -4 * x + 24
    let y_axis : ℝ → ℝ := λ x => 0
    let intersection_x : ℝ := 30 / 7
    let intersection_y : ℝ := line1 intersection_x
    let y_intercept1 : ℝ := line1 0
    let y_intercept2 : ℝ := line2 0
    area = 450 / 7 ∧
    area = (1 / 2) * (y_intercept2 - y_intercept1) * intersection_x

/-- Proof of the triangle area theorem -/
theorem triangle_area_proof : triangle_area (450 / 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2794_279460


namespace NUMINAMATH_CALUDE_divides_condition_l2794_279458

theorem divides_condition (a b : ℕ) : 
  (a^b + b) ∣ (a^(2*b) + 2*b) ↔ 
  (a = 0) ∨ (b = 0) ∨ (a = 2 ∧ b = 1) := by
  sorry

-- Define 0^0 = 1
axiom zero_pow_zero : (0 : ℕ)^(0 : ℕ) = 1

end NUMINAMATH_CALUDE_divides_condition_l2794_279458


namespace NUMINAMATH_CALUDE_cards_lost_ratio_l2794_279465

/-- Represents the number of cards Phil buys each week -/
def cards_per_week : ℕ := 20

/-- Represents the number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- Represents the number of cards Phil has left after the fire -/
def cards_left : ℕ := 520

/-- Theorem stating that the ratio of cards lost to total cards before the fire is 1:2 -/
theorem cards_lost_ratio :
  let total_cards := cards_per_week * weeks_in_year
  let lost_cards := total_cards - cards_left
  (lost_cards : ℚ) / total_cards = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cards_lost_ratio_l2794_279465


namespace NUMINAMATH_CALUDE_equation_solutions_l2794_279468

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - (x - 1) = 7 ∧ x = 3) ∧
  (∃ x : ℚ, (2 * x - 1) / 3 - (x - 3) / 6 = 1 ∧ x = 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2794_279468


namespace NUMINAMATH_CALUDE_five_people_six_chairs_l2794_279442

/-- The number of ways to arrange n people in m chairs -/
def arrange (n : ℕ) (m : ℕ) : ℕ := sorry

/-- There are 5 people and 6 chairs -/
def num_people : ℕ := 5
def num_chairs : ℕ := 6

theorem five_people_six_chairs :
  arrange num_people num_chairs = 720 := by sorry

end NUMINAMATH_CALUDE_five_people_six_chairs_l2794_279442


namespace NUMINAMATH_CALUDE_quadratic_square_completion_l2794_279447

theorem quadratic_square_completion (p q : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 3 = 0 ↔ (x + p)^2 = q) → p + q = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_square_completion_l2794_279447


namespace NUMINAMATH_CALUDE_sector_angle_values_l2794_279471

-- Define a sector
structure Sector where
  radius : ℝ
  centralAngle : ℝ

-- Define the perimeter and area of a sector
def perimeter (s : Sector) : ℝ := 2 * s.radius + s.radius * s.centralAngle
def area (s : Sector) : ℝ := 0.5 * s.radius * s.radius * s.centralAngle

-- Theorem statement
theorem sector_angle_values :
  ∃ s : Sector, perimeter s = 6 ∧ area s = 2 ∧ (s.centralAngle = 1 ∨ s.centralAngle = 4) :=
sorry

end NUMINAMATH_CALUDE_sector_angle_values_l2794_279471


namespace NUMINAMATH_CALUDE_range_of_m_for_increasing_f_l2794_279453

/-- A quadratic function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The property that f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, x ≥ -2 → y ≥ -2 → x < y → f m x < f m y

/-- The theorem stating the range of m for which f is increasing on [-2, +∞) -/
theorem range_of_m_for_increasing_f :
  ∀ m : ℝ, is_increasing_on_interval m → m ≤ -16 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_increasing_f_l2794_279453


namespace NUMINAMATH_CALUDE_kaleb_boxes_correct_l2794_279483

/-- The number of boxes Kaleb bought initially -/
def initial_boxes : ℕ := 9

/-- The number of boxes Kaleb gave to his little brother -/
def given_boxes : ℕ := 5

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 6

/-- The number of pieces Kaleb still has -/
def remaining_pieces : ℕ := 54

/-- Theorem stating that the initial number of boxes is correct -/
theorem kaleb_boxes_correct :
  initial_boxes * pieces_per_box = remaining_pieces + given_boxes * pieces_per_box :=
by sorry

end NUMINAMATH_CALUDE_kaleb_boxes_correct_l2794_279483


namespace NUMINAMATH_CALUDE_function_difference_implies_m_value_l2794_279400

/-- The function f(x) = 4x^2 - 3x + 5 -/
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5

/-- The function g(x) = x^2 - mx - 8, parameterized by m -/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x - 8

/-- Theorem stating that if f(5) - g(5) = 15, then m = -11.6 -/
theorem function_difference_implies_m_value :
  ∃ m : ℝ, f 5 - g m 5 = 15 → m = -11.6 := by
  sorry

end NUMINAMATH_CALUDE_function_difference_implies_m_value_l2794_279400


namespace NUMINAMATH_CALUDE_inequality_holds_l2794_279457

theorem inequality_holds (r s : ℝ) (hr : 0 ≤ r ∧ r < 2) (hs : s > 0) :
  (4 * (r * s^2 + r^2 * s + 4 * s^2 + 4 * r * s)) / (r + s) > 3 * r^2 * s := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l2794_279457


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l2794_279407

/-- Given 6 people in an elevator with an average weight of 152 lbs, 
    prove that if a 7th person enters and the new average becomes 151 lbs, 
    then the weight of the 7th person is 145 lbs. -/
theorem elevator_weight_problem (people : ℕ) (avg_weight_before : ℝ) (avg_weight_after : ℝ) :
  people = 6 →
  avg_weight_before = 152 →
  avg_weight_after = 151 →
  (people * avg_weight_before + (avg_weight_after * (people + 1) - people * avg_weight_before)) = 145 :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l2794_279407


namespace NUMINAMATH_CALUDE_cosine_sum_less_than_sum_of_cosines_l2794_279401

theorem cosine_sum_less_than_sum_of_cosines (α β : Real) 
  (h_α : 0 < α ∧ α < π / 2) (h_β : 0 < β ∧ β < π / 2) : 
  Real.cos (α + β) < Real.cos α + Real.cos β := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_less_than_sum_of_cosines_l2794_279401


namespace NUMINAMATH_CALUDE_initial_loss_percentage_l2794_279456

-- Define the cost price of a pencil
def cost_price : ℚ := 1 / 13

-- Define the selling price when selling 20 pencils for 1 rupee
def selling_price_20 : ℚ := 1 / 20

-- Define the selling price when selling 10 pencils for 1 rupee (30% gain)
def selling_price_10 : ℚ := 1 / 10

-- Define the percentage loss
def percentage_loss : ℚ := ((cost_price - selling_price_20) / cost_price) * 100

-- Theorem stating the initial loss percentage
theorem initial_loss_percentage : 
  (selling_price_10 = cost_price + 0.3 * cost_price) → 
  (percentage_loss = 35) :=
by
  sorry

end NUMINAMATH_CALUDE_initial_loss_percentage_l2794_279456


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2794_279487

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2 * x - 1 = 0 ∧ m * y^2 + 2 * y - 1 = 0) ↔ 
  (m > -1 ∧ m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2794_279487


namespace NUMINAMATH_CALUDE_family_ages_l2794_279467

theorem family_ages (man son daughter : ℕ) : 
  man = son + 46 →
  man + 2 = 2 * (son + 2) →
  daughter = son - 4 →
  son + daughter = 84 := by
sorry

end NUMINAMATH_CALUDE_family_ages_l2794_279467


namespace NUMINAMATH_CALUDE_fraction_decimal_digits_l2794_279472

/-- The fraction we're considering -/
def fraction : ℚ := 987654321 / (2^30 * 5^2 * 3)

/-- The minimum number of digits to the right of the decimal point -/
def min_decimal_digits : ℕ := 30

/-- Theorem stating that the minimum number of digits to the right of the decimal point
    needed to express the fraction as a decimal is equal to min_decimal_digits -/
theorem fraction_decimal_digits :
  (∀ n : ℕ, n < min_decimal_digits → ∃ m : ℕ, fraction * 10^n ≠ m) ∧
  (∃ m : ℕ, fraction * 10^min_decimal_digits = m) :=
sorry

end NUMINAMATH_CALUDE_fraction_decimal_digits_l2794_279472


namespace NUMINAMATH_CALUDE_select_parts_with_first_class_l2794_279491

theorem select_parts_with_first_class (total : Nat) (first_class : Nat) (second_class : Nat) (select : Nat) :
  total = first_class + second_class →
  first_class = 5 →
  second_class = 3 →
  select = 3 →
  (Nat.choose total select) - (Nat.choose second_class select) = 55 := by
  sorry

end NUMINAMATH_CALUDE_select_parts_with_first_class_l2794_279491


namespace NUMINAMATH_CALUDE_point_outside_circle_l2794_279479

/-- A circle with a given diameter -/
structure Circle where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- A point with a given distance from the center of a circle -/
structure Point (c : Circle) where
  distance_from_center : ℝ
  distance_pos : distance_from_center > 0

/-- Definition of a point being outside a circle -/
def is_outside (c : Circle) (p : Point c) : Prop :=
  p.distance_from_center > c.diameter / 2

theorem point_outside_circle (c : Circle) (p : Point c) 
  (h_diam : c.diameter = 10) 
  (h_dist : p.distance_from_center = 6) : 
  is_outside c p := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2794_279479


namespace NUMINAMATH_CALUDE_fourth_power_sum_equality_l2794_279446

theorem fourth_power_sum_equality : 120^4 + 97^4 + 84^4 + 27^4 = 174^4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_equality_l2794_279446


namespace NUMINAMATH_CALUDE_binary_operation_proof_l2794_279420

/-- Convert a binary number (represented as a list of bits) to a natural number -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- The first binary number 11001₂ -/
def num1 : List Bool := [true, true, false, false, true]

/-- The second binary number 1101₂ -/
def num2 : List Bool := [true, true, false, true]

/-- The third binary number 101₂ -/
def num3 : List Bool := [true, false, true]

/-- The result 100111010₂ -/
def result : List Bool := [true, false, false, true, true, true, false, true, false]

/-- Theorem stating that (11001₂ * 1101₂) - 101₂ = 100111010₂ -/
theorem binary_operation_proof :
  (binary_to_nat num1 * binary_to_nat num2) - binary_to_nat num3 = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_binary_operation_proof_l2794_279420


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_example_l2794_279495

/-- A parallelogram with side lengths a and b -/
structure Parallelogram where
  a : ℝ
  b : ℝ

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := 2 * (p.a + p.b)

theorem parallelogram_perimeter_example : 
  let p : Parallelogram := { a := 10, b := 7 }
  perimeter p = 34 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_example_l2794_279495


namespace NUMINAMATH_CALUDE_smallest_t_for_no_h_route_l2794_279425

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)
  (h_size : size = 8)

/-- Represents a Horse's move --/
structure HorseMove :=
  (horizontal : Nat)
  (vertical : Nat)

/-- Represents an H Route --/
def HRoute (board : Chessboard) (move : HorseMove) : Prop :=
  ∃ (path : List (Nat × Nat)), 
    path.length = board.size * board.size ∧
    ∀ (pos : Nat × Nat), pos ∈ path → 
      pos.1 ≤ board.size ∧ pos.2 ≤ board.size

/-- The main theorem --/
theorem smallest_t_for_no_h_route : 
  ∀ (board : Chessboard),
    ∀ (t : Nat),
      t > 0 →
      (∀ (start : Nat × Nat), 
        start.1 ≤ board.size ∧ start.2 ≤ board.size →
        ¬ HRoute board ⟨t, t+1⟩) →
      t = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_t_for_no_h_route_l2794_279425


namespace NUMINAMATH_CALUDE_geralds_initial_notebooks_l2794_279463

theorem geralds_initial_notebooks (jack_initial gerald_initial jack_remaining paula_given mike_given : ℕ) : 
  jack_initial = gerald_initial + 13 →
  jack_initial = jack_remaining + paula_given + mike_given →
  jack_remaining = 10 →
  paula_given = 5 →
  mike_given = 6 →
  gerald_initial = 8 := by
sorry

end NUMINAMATH_CALUDE_geralds_initial_notebooks_l2794_279463


namespace NUMINAMATH_CALUDE_nearest_whole_number_to_24567_4999997_l2794_279452

theorem nearest_whole_number_to_24567_4999997 :
  let x : ℝ := 24567.4999997
  ∃ (n : ℤ), ∀ (m : ℤ), |x - n| ≤ |x - m| ∧ n = 24567 :=
sorry

end NUMINAMATH_CALUDE_nearest_whole_number_to_24567_4999997_l2794_279452


namespace NUMINAMATH_CALUDE_number_difference_l2794_279421

theorem number_difference (x y : ℝ) : 
  (35 + x) / 2 = 45 →
  (35 + x + y) / 3 = 40 →
  |y - 35| = 5 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2794_279421


namespace NUMINAMATH_CALUDE_teachers_in_school_l2794_279416

/-- Calculates the number of teachers required in a school --/
def teachers_required (total_students : ℕ) (lessons_per_student : ℕ) (lessons_per_teacher : ℕ) (students_per_class : ℕ) : ℕ :=
  (total_students * lessons_per_student) / (students_per_class * lessons_per_teacher)

/-- Theorem stating that 50 teachers are required given the specific conditions --/
theorem teachers_in_school : 
  teachers_required 1200 5 4 30 = 50 := by
  sorry

#eval teachers_required 1200 5 4 30

end NUMINAMATH_CALUDE_teachers_in_school_l2794_279416


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2794_279464

-- Define set A
def A : Set ℝ := {y | ∃ x, y = Real.exp x}

-- Define set B
def B : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2794_279464


namespace NUMINAMATH_CALUDE_heath_carrot_planting_l2794_279418

/-- Given the conditions of Heath's carrot planting, prove the number of plants in each row. -/
theorem heath_carrot_planting 
  (total_rows : ℕ) 
  (planting_time : ℕ) 
  (planting_rate : ℕ) 
  (h1 : total_rows = 400)
  (h2 : planting_time = 20)
  (h3 : planting_rate = 6000) :
  (planting_time * planting_rate) / total_rows = 300 := by
  sorry

end NUMINAMATH_CALUDE_heath_carrot_planting_l2794_279418


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l2794_279493

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the given tetrahedron is approximately 13.416 -/
theorem volume_of_specific_tetrahedron :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := 3,
    RS := 7
  }
  abs (tetrahedronVolume t - 13.416) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l2794_279493


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2794_279496

theorem quadratic_roots_properties (x₁ x₂ : ℝ) :
  x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0 →
  (x₁ + x₂) * (x₁ * x₂) = -2 ∧ (x₁ - x₂)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2794_279496


namespace NUMINAMATH_CALUDE_children_on_bus_after_stop_l2794_279469

theorem children_on_bus_after_stop (initial : ℕ) (got_on : ℕ) (got_off : ℕ) :
  initial = 22 → got_on = 40 → got_off = 60 →
  initial + got_on - got_off = 2 := by
  sorry

end NUMINAMATH_CALUDE_children_on_bus_after_stop_l2794_279469


namespace NUMINAMATH_CALUDE_gcd_division_remainder_l2794_279429

theorem gcd_division_remainder (a b : ℕ) (h1 : a > b) (h2 : ∃ q r : ℕ, a = b * q + r ∧ 0 < r ∧ r < b) :
  Nat.gcd a b = Nat.gcd b (a % b) :=
by sorry

end NUMINAMATH_CALUDE_gcd_division_remainder_l2794_279429


namespace NUMINAMATH_CALUDE_vertically_opposite_angles_equal_l2794_279459

-- Define a type for angles
def Angle : Type := ℝ

-- Define a function to represent vertically opposite angles
def verticallyOpposite (α β : Angle) : Prop := sorry

-- Theorem: Vertically opposite angles are equal
theorem vertically_opposite_angles_equal (α β : Angle) :
  verticallyOpposite α β → α = β :=
sorry

end NUMINAMATH_CALUDE_vertically_opposite_angles_equal_l2794_279459


namespace NUMINAMATH_CALUDE_largest_choir_size_l2794_279475

theorem largest_choir_size : 
  ∃ (n : ℕ), 
    (∃ (k : ℕ), n = k^2 + 11) ∧ 
    (∃ (m : ℕ), n = m * (m + 5)) ∧
    (∀ (x : ℕ), 
      ((∃ (k : ℕ), x = k^2 + 11) ∧ 
       (∃ (m : ℕ), x = m * (m + 5))) → 
      x ≤ n) ∧
    n = 325 :=
by sorry

end NUMINAMATH_CALUDE_largest_choir_size_l2794_279475


namespace NUMINAMATH_CALUDE_equation_solution_l2794_279406

theorem equation_solution : 
  ∃ x : ℝ, (x / (x - 1) - 1 = 1) ∧ (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2794_279406


namespace NUMINAMATH_CALUDE_proposition_c_is_true_l2794_279408

theorem proposition_c_is_true : ∀ x y : ℝ, x + y ≠ 3 → x ≠ 2 ∨ y ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_proposition_c_is_true_l2794_279408


namespace NUMINAMATH_CALUDE_divisibility_of_p_and_q_l2794_279433

def ones (n : ℕ) : ℕ := (10^n - 1) / 9

def p (n : ℕ) : ℕ := ones n * (10^(3*n) + 9*10^(2*n) + 8*10^n + 7)

def q (n : ℕ) : ℕ := ones (n+1) * (10^(3*(n+1)) + 9*10^(2*(n+1)) + 8*10^(n+1) + 7)

theorem divisibility_of_p_and_q (n : ℕ) (h : 1987 ∣ ones n) : 
  1987 ∣ p n ∧ 1987 ∣ q n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_p_and_q_l2794_279433


namespace NUMINAMATH_CALUDE_sum_of_variables_l2794_279445

theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 10 - 2*x)
  (eq2 : x + z = -12 - 4*y)
  (eq3 : x + y = 5 - 2*z) :
  2*x + 2*y + 2*z = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l2794_279445


namespace NUMINAMATH_CALUDE_kennel_arrangement_count_l2794_279431

/-- The number of chickens in the kennel -/
def num_chickens : Nat := 4

/-- The number of dogs in the kennel -/
def num_dogs : Nat := 3

/-- The number of cats in the kennel -/
def num_cats : Nat := 5

/-- The total number of animals in the kennel -/
def total_animals : Nat := num_chickens + num_dogs + num_cats

/-- The number of ways to arrange animals within their groups -/
def intra_group_arrangements : Nat := (Nat.factorial num_chickens) * (Nat.factorial num_dogs) * (Nat.factorial num_cats)

/-- The number of valid group orders (chickens-dogs-cats and chickens-cats-dogs) -/
def valid_group_orders : Nat := 2

/-- The total number of ways to arrange the animals -/
def total_arrangements : Nat := valid_group_orders * intra_group_arrangements

theorem kennel_arrangement_count :
  total_arrangements = 34560 :=
sorry

end NUMINAMATH_CALUDE_kennel_arrangement_count_l2794_279431


namespace NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l2794_279497

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ
  | 0 => a₁
  | k + 1 => a₁ + k * d

theorem third_term_of_arithmetic_sequence :
  let a₁ : ℝ := 11
  let a₆ : ℝ := 39
  let n : ℕ := 6
  let d : ℝ := (a₆ - a₁) / (n - 1)
  arithmetic_sequence a₁ d n 2 = 22.2 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l2794_279497


namespace NUMINAMATH_CALUDE_vector_sum_parallel_l2794_279430

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem vector_sum_parallel (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → a.1 + b.1 = -2 ∧ a.2 + b.2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_parallel_l2794_279430


namespace NUMINAMATH_CALUDE_garden_flowers_l2794_279432

theorem garden_flowers (white_flowers : ℕ) (additional_red_needed : ℕ) (current_red_flowers : ℕ) : 
  white_flowers = 555 →
  additional_red_needed = 208 →
  white_flowers = current_red_flowers + additional_red_needed →
  current_red_flowers = 347 := by
sorry

end NUMINAMATH_CALUDE_garden_flowers_l2794_279432


namespace NUMINAMATH_CALUDE_number_of_students_l2794_279455

theorem number_of_students (total_skittles : ℕ) (skittles_per_student : ℕ) (h1 : total_skittles = 27) (h2 : skittles_per_student = 3) :
  total_skittles / skittles_per_student = 9 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_l2794_279455


namespace NUMINAMATH_CALUDE_unique_solution_l2794_279482

theorem unique_solution : ∀ a b : ℕ+,
  (¬ (7 ∣ (a * b * (a + b)))) →
  ((7^7) ∣ ((a + b)^7 - a^7 - b^7)) →
  (a = 18 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2794_279482


namespace NUMINAMATH_CALUDE_exactly_three_imply_l2794_279474

open Classical

variables (p q r : Prop)

def statement1 : Prop := ¬p ∧ q ∧ ¬r
def statement2 : Prop := p ∧ ¬q ∧ ¬r
def statement3 : Prop := ¬p ∧ ¬q ∧ r
def statement4 : Prop := p ∧ q ∧ ¬r

def implication : Prop := (¬p → ¬q) → ¬r

theorem exactly_three_imply :
  ∃! (n : Nat), n = 3 ∧
  (n = (if statement1 p q r → implication p q r then 1 else 0) +
       (if statement2 p q r → implication p q r then 1 else 0) +
       (if statement3 p q r → implication p q r then 1 else 0) +
       (if statement4 p q r → implication p q r then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_three_imply_l2794_279474


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_l2794_279409

theorem min_bottles_to_fill (small_capacity large_capacity : ℕ) 
  (h1 : small_capacity = 40)
  (h2 : large_capacity = 360) : 
  Nat.ceil (large_capacity / small_capacity) = 9 := by
  sorry

#check min_bottles_to_fill

end NUMINAMATH_CALUDE_min_bottles_to_fill_l2794_279409


namespace NUMINAMATH_CALUDE_range_of_a_l2794_279423

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then x^2 - 2 else x + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  f a (Real.log x + 1/x) - a

theorem range_of_a (a : ℝ) :
  (∃ x > 0, g a x = 0) → a ∈ Set.Icc (-1) 2 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2794_279423


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2794_279419

theorem two_numbers_difference (x y : ℤ) 
  (sum_eq : x + y = 40)
  (triple_minus_double : 3 * max x y - 2 * min x y = 8) :
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2794_279419


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_specific_a_value_l2794_279411

/-- Hyperbola C: x²/a² - y² = 1 (a > 0) -/
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1 ∧ a > 0

/-- Line l: x + y = 1 -/
def line (x y : ℝ) : Prop := x + y = 1

/-- P is the intersection point of l and the y-axis -/
def P : ℝ × ℝ := (0, 1)

/-- A and B are distinct intersection points of C and l -/
def intersection_points (a : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧
    hyperbola a A.1 A.2 ∧ line A.1 A.2 ∧
    hyperbola a B.1 B.2 ∧ line B.1 B.2

/-- PA = (5/12)PB -/
def vector_relation (A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (5/12 * (B.1 - P.1), 5/12 * (B.2 - P.2))

theorem hyperbola_line_intersection (a : ℝ) :
  intersection_points a → (0 < a ∧ a < 1) ∨ (1 < a ∧ a < Real.sqrt 2) :=
sorry

theorem specific_a_value (a : ℝ) (A B : ℝ × ℝ) :
  hyperbola a A.1 A.2 ∧ line A.1 A.2 ∧
  hyperbola a B.1 B.2 ∧ line B.1 B.2 ∧
  vector_relation A B →
  a = 17/13 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_specific_a_value_l2794_279411


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l2794_279422

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 3 * n^2 + 4 * n + 5

/-- The r-th term of the arithmetic progression -/
def a (r : ℕ) : ℝ := 6 * r + 1

theorem arithmetic_progression_rth_term (r : ℕ) : 
  a r = S r - S (r - 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l2794_279422


namespace NUMINAMATH_CALUDE_travel_options_count_l2794_279434

/-- The number of flights from A to B in one day -/
def num_flights : ℕ := 3

/-- The number of trains from A to B in one day -/
def num_trains : ℕ := 2

/-- The total number of ways to travel from A to B in one day -/
def total_ways : ℕ := num_flights + num_trains

theorem travel_options_count : total_ways = 5 := by sorry

end NUMINAMATH_CALUDE_travel_options_count_l2794_279434


namespace NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l2794_279440

/-- The number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries needed to trade for one zucchini -/
def containers_per_zucchini : ℚ := 4 / 3

/-- The number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 72

/-- The minimum number of bushes needed to obtain at least the target number of zucchinis -/
def min_bushes_needed : ℕ :=
  (target_zucchinis * containers_per_zucchini / containers_per_bush).ceil.toNat

theorem min_bushes_for_zucchinis :
  min_bushes_needed = 10 := by sorry

end NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l2794_279440


namespace NUMINAMATH_CALUDE_option_c_is_experimental_l2794_279443

-- Define a type for survey methods
inductive SurveyMethod
| Direct
| Experimental
| SecondaryData

-- Define a type for survey options
inductive SurveyOption
| A
| B
| C
| D

-- Define a function that assigns a survey method to each option
def survey_method (option : SurveyOption) : SurveyMethod :=
  match option with
  | SurveyOption.A => SurveyMethod.Direct
  | SurveyOption.B => SurveyMethod.Direct
  | SurveyOption.C => SurveyMethod.Experimental
  | SurveyOption.D => SurveyMethod.SecondaryData

-- Define the experimental method suitability
def is_suitable_for_experimental (method : SurveyMethod) : Prop :=
  method = SurveyMethod.Experimental

-- Theorem: Option C is the only one suitable for the experimental method
theorem option_c_is_experimental :
  ∀ (option : SurveyOption),
    is_suitable_for_experimental (survey_method option) ↔ option = SurveyOption.C :=
by
  sorry

#check option_c_is_experimental

end NUMINAMATH_CALUDE_option_c_is_experimental_l2794_279443


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l2794_279498

/-- Calculates the time it takes for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 10)
  (h2 : train_speed = 46)
  (h3 : train_length = 120)
  (h4 : initial_distance = 340)
  : (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 46 := by
  sorry

#check train_passing_jogger_time

end NUMINAMATH_CALUDE_train_passing_jogger_time_l2794_279498


namespace NUMINAMATH_CALUDE_new_boarders_count_new_boarders_joined_school_l2794_279426

theorem new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders : ℕ) (initial_ratio_day : ℕ)
                            (new_ratio_boarders : ℕ) (new_ratio_day : ℕ) : ℕ :=
  let initial_day_students := initial_boarders * initial_ratio_day / initial_ratio_boarders
  let new_boarders := initial_day_students * new_ratio_boarders / new_ratio_day - initial_boarders
  new_boarders

theorem new_boarders_joined_school :
  new_boarders_count 60 2 5 1 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_new_boarders_count_new_boarders_joined_school_l2794_279426


namespace NUMINAMATH_CALUDE_count_non_dividing_eq_29_l2794_279450

/-- g(n) is the product of proper positive integer divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- count_non_dividing counts the number of integers n between 2 and 100 (inclusive) 
    for which n does not divide g(n) -/
def count_non_dividing : ℕ := sorry

/-- Theorem stating that the count of integers n between 2 and 100 (inclusive) 
    for which n does not divide g(n) is equal to 29 -/
theorem count_non_dividing_eq_29 : count_non_dividing = 29 := by sorry

end NUMINAMATH_CALUDE_count_non_dividing_eq_29_l2794_279450


namespace NUMINAMATH_CALUDE_quadratic_zero_discriminant_l2794_279449

/-- The quadratic equation 5x^2 - 10x√3 + k = 0 has zero discriminant if and only if k = 15 -/
theorem quadratic_zero_discriminant (k : ℝ) :
  (∀ x : ℝ, 5 * x^2 - 10 * x * Real.sqrt 3 + k = 0) →
  ((-10 * Real.sqrt 3)^2 - 4 * 5 * k = 0) ↔
  k = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_zero_discriminant_l2794_279449


namespace NUMINAMATH_CALUDE_max_profit_min_sales_for_profit_l2794_279436

-- Define the cost per unit
def cost : ℝ := 20

-- Define the relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -10 * x + 500

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost) * sales_volume x

-- Define the price constraints
def price_constraint (x : ℝ) : Prop := 25 ≤ x ∧ x ≤ 38

-- Theorem 1: Maximum profit occurs at x = 35 and is equal to 2250
theorem max_profit :
  ∃ (x : ℝ), price_constraint x ∧
  profit x = 2250 ∧
  ∀ (y : ℝ), price_constraint y → profit y ≤ profit x :=
sorry

-- Theorem 2: At price 38, selling 120 units yields a profit of at least 2000
theorem min_sales_for_profit :
  sales_volume 38 ≥ 120 ∧ profit 38 ≥ 2000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_min_sales_for_profit_l2794_279436


namespace NUMINAMATH_CALUDE_odd_function_derivative_l2794_279413

theorem odd_function_derivative (f : ℝ → ℝ) (x₀ : ℝ) (k : ℝ) :
  (∀ x, f (-x) = -f x) →
  Differentiable ℝ f →
  deriv f (-x₀) = k →
  k ≠ 0 →
  deriv f x₀ = k :=
by sorry

end NUMINAMATH_CALUDE_odd_function_derivative_l2794_279413


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l2794_279466

theorem isosceles_right_triangle_hypotenuse (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 2 →
  4 * (1/2 * triangle_leg^2) = square_side^2 →
  triangle_leg^2 + triangle_leg^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l2794_279466


namespace NUMINAMATH_CALUDE_limit_S_over_a_squared_ln_a_nonzero_l2794_279490

/-- The area S(a) bounded by the curve y = (a-x)ln x and the x-axis for a > 1 -/
noncomputable def S (a : ℝ) : ℝ := ∫ x in (1)..(a), (a - x) * Real.log x

/-- The limit of S(a)/(a^2 ln a) as a approaches infinity is a non-zero real number -/
theorem limit_S_over_a_squared_ln_a_nonzero :
  ∃ (L : ℝ), L ≠ 0 ∧ Filter.Tendsto (fun a => S a / (a^2 * Real.log a)) Filter.atTop (nhds L) := by
  sorry

end NUMINAMATH_CALUDE_limit_S_over_a_squared_ln_a_nonzero_l2794_279490


namespace NUMINAMATH_CALUDE_rod_cutting_l2794_279477

theorem rod_cutting (rod_length : ℝ) (total_pieces : ℝ) (piece_length : ℝ) : 
  rod_length = 47.5 →
  total_pieces = 118.75 →
  piece_length = rod_length / total_pieces →
  piece_length = 0.4 := by
sorry

end NUMINAMATH_CALUDE_rod_cutting_l2794_279477


namespace NUMINAMATH_CALUDE_no_integer_roots_l2794_279486

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluation of a polynomial at a point -/
def eval (P : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

/-- A number is odd if it's not divisible by 2 -/
def IsOdd (n : ℤ) : Prop := n % 2 ≠ 0

theorem no_integer_roots (P : IntPolynomial) 
  (h0 : IsOdd (eval P 0)) 
  (h1 : IsOdd (eval P 1)) : 
  ∀ (n : ℤ), eval P n ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l2794_279486


namespace NUMINAMATH_CALUDE_power_of_two_not_sum_of_consecutive_integers_l2794_279403

theorem power_of_two_not_sum_of_consecutive_integers :
  ∀ n : ℕ+, (∀ r : ℕ, r > 1 → ¬∃ k : ℕ, n = (k + r) * (k + r - 1) / 2 - k * (k - 1) / 2) ↔
  ∃ l : ℕ, n = 2^l := by sorry

end NUMINAMATH_CALUDE_power_of_two_not_sum_of_consecutive_integers_l2794_279403


namespace NUMINAMATH_CALUDE_prob_at_least_two_white_correct_l2794_279485

/-- The probability of drawing at least two white balls in three draws from a bag 
    containing 2 red balls and 4 white balls, with replacement -/
def prob_at_least_two_white : ℚ := 20 / 27

/-- The total number of balls in the bag -/
def total_balls : ℕ := 6

/-- The number of white balls in the bag -/
def white_balls : ℕ := 4

/-- The number of draws -/
def num_draws : ℕ := 3

theorem prob_at_least_two_white_correct : 
  prob_at_least_two_white = 
    (Nat.choose num_draws 2 * (white_balls / total_balls)^2 * ((total_balls - white_balls) / total_balls)) +
    (white_balls / total_balls)^num_draws :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_two_white_correct_l2794_279485


namespace NUMINAMATH_CALUDE_equal_roots_sum_inverse_a_and_c_l2794_279454

theorem equal_roots_sum_inverse_a_and_c (a c : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, x * x * a + 2 * x + 2 - c = 0 ∧ 
   ∀ y : ℝ, y * y * a + 2 * y + 2 - c = 0 → y = x) →
  1 / a + c = 2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_sum_inverse_a_and_c_l2794_279454


namespace NUMINAMATH_CALUDE_g10_diamonds_l2794_279427

/-- Number of diamonds in figure G_n -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 10
  else 2 + 2 * n^2 + 2 * n - 4

/-- The sequence of figures G_n satisfies the given properties -/
axiom sequence_property (n : ℕ) (h : n ≥ 3) :
  num_diamonds n = num_diamonds (n - 1) + 4 * (n + 1)

/-- G_1 has 2 diamonds -/
axiom g1_diamonds : num_diamonds 1 = 2

/-- G_2 has 10 diamonds -/
axiom g2_diamonds : num_diamonds 2 = 10

/-- Theorem: G_10 has 218 diamonds -/
theorem g10_diamonds : num_diamonds 10 = 218 := by sorry

end NUMINAMATH_CALUDE_g10_diamonds_l2794_279427
