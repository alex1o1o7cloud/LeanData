import Mathlib

namespace NUMINAMATH_CALUDE_rice_mixture_price_l3643_364321

/-- Represents the price of rice in Rupees per kilogram -/
@[ext] structure RicePrice where
  price : ℝ

/-- Represents a mixture of two types of rice -/
structure RiceMixture where
  price1 : RicePrice
  price2 : RicePrice
  ratio : ℝ
  mixtureCost : ℝ

/-- The theorem statement -/
theorem rice_mixture_price (mix : RiceMixture) 
  (h1 : mix.price1.price = 16)
  (h2 : mix.ratio = 3)
  (h3 : mix.mixtureCost = 18) :
  mix.price2.price = 24 := by
  sorry

end NUMINAMATH_CALUDE_rice_mixture_price_l3643_364321


namespace NUMINAMATH_CALUDE_kaydence_age_l3643_364389

/-- The age of Kaydence given the ages of her family members and the total family age -/
theorem kaydence_age (total_age father_age mother_age brother_age sister_age : ℕ)
  (h_total : total_age = 200)
  (h_father : father_age = 60)
  (h_mother : mother_age = father_age - 2)
  (h_brother : brother_age = father_age / 2)
  (h_sister : sister_age = 40) :
  total_age - (father_age + mother_age + brother_age + sister_age) = 12 := by
  sorry

end NUMINAMATH_CALUDE_kaydence_age_l3643_364389


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3643_364376

theorem trigonometric_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.cos y * Real.sin (x + y) = Real.cos y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3643_364376


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l3643_364306

theorem reciprocal_sum_pairs : 
  ∃! k : ℕ, k > 0 ∧ 
  (∃ S : Finset (ℕ × ℕ), 
    (∀ (m n : ℕ), (m, n) ∈ S ↔ m > 0 ∧ n > 0 ∧ 1 / m + 1 / n = 1 / 5) ∧
    Finset.card S = k) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l3643_364306


namespace NUMINAMATH_CALUDE_correct_quotient_calculation_l3643_364371

theorem correct_quotient_calculation (dividend : ℕ) (incorrect_quotient : ℕ) : 
  dividend > 0 →
  incorrect_quotient = 753 →
  dividend = 102 * (incorrect_quotient * 3) →
  dividend % 201 = 0 →
  dividend / 201 = 1146 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_calculation_l3643_364371


namespace NUMINAMATH_CALUDE_proposition_implication_l3643_364348

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k ≥ 1 → (P k → P (k + 1)))
  (h2 : ¬ P 10) : 
  ¬ P 9 := by sorry

end NUMINAMATH_CALUDE_proposition_implication_l3643_364348


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l3643_364332

/-- Calculates the final weight after a two-week diet plan -/
def final_weight (initial_weight : ℝ) (first_week_loss : ℝ) (second_week_rate : ℝ) : ℝ :=
  initial_weight - (first_week_loss + second_week_rate * first_week_loss)

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss :
  let initial_weight : ℝ := 92
  let first_week_loss : ℝ := 5
  let second_week_rate : ℝ := 1.3
  final_weight initial_weight first_week_loss second_week_rate = 80.5 := by
  sorry

#eval final_weight 92 5 1.3

end NUMINAMATH_CALUDE_jessie_weight_loss_l3643_364332


namespace NUMINAMATH_CALUDE_children_going_to_zoo_l3643_364390

/-- The number of children per seat in the bus -/
def children_per_seat : ℕ := 2

/-- The total number of seats needed in the bus -/
def total_seats : ℕ := 29

/-- The total number of children taking the bus to the zoo -/
def total_children : ℕ := children_per_seat * total_seats

theorem children_going_to_zoo : total_children = 58 := by
  sorry

end NUMINAMATH_CALUDE_children_going_to_zoo_l3643_364390


namespace NUMINAMATH_CALUDE_quadratic_polynomial_bound_l3643_364310

/-- A polynomial of degree 2 with real, nonnegative coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c

/-- The value of the quadratic polynomial at a given x -/
def QuadraticPolynomial.eval (g : QuadraticPolynomial) (x : ℝ) : ℝ :=
  g.a * x^2 + g.b * x + g.c

theorem quadratic_polynomial_bound (g : QuadraticPolynomial) 
  (h1 : g.eval 3 = 3) (h2 : g.eval 9 = 243) : 
  g.eval 6 ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_bound_l3643_364310


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l3643_364388

/-- Calculates the actual percent profit when a shopkeeper labels an item's price
    to earn a specified profit percentage and then offers a discount. -/
def actualPercentProfit (labeledProfitPercent : ℝ) (discountPercent : ℝ) : ℝ :=
  let labeledPrice := 1 + labeledProfitPercent
  let sellingPrice := labeledPrice * (1 - discountPercent)
  (sellingPrice - 1) * 100

/-- Proves that when a shopkeeper labels an item's price to earn a 30% profit
    on the cost price and then offers a 10% discount on the labeled price,
    the actual percent profit earned is 17%. -/
theorem shopkeeper_profit :
  actualPercentProfit 0.3 0.1 = 17 :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l3643_364388


namespace NUMINAMATH_CALUDE_yoongi_stack_higher_l3643_364379

/-- The height of Box A in centimeters -/
def box_a_height : ℝ := 3

/-- The height of Box B in centimeters -/
def box_b_height : ℝ := 3.5

/-- The number of Box A stacked by Taehyung -/
def taehyung_boxes : ℕ := 16

/-- The number of Box B stacked by Yoongi -/
def yoongi_boxes : ℕ := 14

/-- The total height of Taehyung's stack in centimeters -/
def taehyung_stack_height : ℝ := box_a_height * taehyung_boxes

/-- The total height of Yoongi's stack in centimeters -/
def yoongi_stack_height : ℝ := box_b_height * yoongi_boxes

theorem yoongi_stack_higher :
  yoongi_stack_height > taehyung_stack_height ∧
  yoongi_stack_height - taehyung_stack_height = 1 :=
by sorry

end NUMINAMATH_CALUDE_yoongi_stack_higher_l3643_364379


namespace NUMINAMATH_CALUDE_eliza_numbers_l3643_364377

theorem eliza_numbers (a b : ℤ) (h1 : 2 * a + 3 * b = 110) (h2 : a = 32 ∨ b = 32) : 
  (a = 7 ∧ b = 32) ∨ (a = 32 ∧ b = 7) := by
sorry

end NUMINAMATH_CALUDE_eliza_numbers_l3643_364377


namespace NUMINAMATH_CALUDE_find_x_l3643_364375

theorem find_x : ∃ x : ℝ, 121 * x = 75625 ∧ x = 625 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3643_364375


namespace NUMINAMATH_CALUDE_right_triangle_circle_theorem_l3643_364360

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle with vertices P, Q, and R -/
structure Triangle :=
  (P : Point)
  (Q : Point)
  (R : Point)

/-- Checks if a triangle is right-angled at Q -/
def isRightTriangle (t : Triangle) : Prop :=
  -- Definition of right triangle at Q
  sorry

/-- Checks if a point S lies on the circle with diameter QR -/
def isOnCircle (t : Triangle) (S : Point) : Prop :=
  -- Definition of S being on the circle with diameter QR
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  -- Definition of distance between two points
  sorry

/-- Main theorem -/
theorem right_triangle_circle_theorem (t : Triangle) (S : Point) :
  isRightTriangle t →
  isOnCircle t S →
  distance t.P S = 3 →
  distance t.Q S = 9 →
  distance t.R S = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circle_theorem_l3643_364360


namespace NUMINAMATH_CALUDE_johns_starting_elevation_l3643_364387

def starting_elevation (rate : ℝ) (time : ℝ) (final_elevation : ℝ) : ℝ :=
  final_elevation + rate * time

theorem johns_starting_elevation :
  starting_elevation 10 5 350 = 400 := by sorry

end NUMINAMATH_CALUDE_johns_starting_elevation_l3643_364387


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3643_364345

theorem parallel_vectors_k_value (k : ℝ) :
  let a : Fin 2 → ℝ := ![k, Real.sqrt 2]
  let b : Fin 2 → ℝ := ![2, -2]
  (∃ (c : ℝ), a = c • b) →
  k = -Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3643_364345


namespace NUMINAMATH_CALUDE_star_neg_two_three_l3643_364315

/-- The "star" operation for rational numbers -/
def star (a b : ℚ) : ℚ := a * b^2 + a

/-- Theorem: The result of (-2)☆3 is -20 -/
theorem star_neg_two_three : star (-2) 3 = -20 := by
  sorry

end NUMINAMATH_CALUDE_star_neg_two_three_l3643_364315


namespace NUMINAMATH_CALUDE_milk_butterfat_calculation_l3643_364326

/-- Represents the butterfat percentage as a real number between 0 and 100 -/
def ButterfatPercentage := { x : ℝ // 0 ≤ x ∧ x ≤ 100 }

/-- Calculates the initial butterfat percentage of milk given the conditions -/
def initial_butterfat_percentage (
  initial_volume : ℝ) 
  (cream_volume : ℝ) 
  (cream_butterfat : ButterfatPercentage) 
  (final_butterfat : ButterfatPercentage) : ButterfatPercentage :=
  sorry

theorem milk_butterfat_calculation :
  let initial_volume : ℝ := 1000
  let cream_volume : ℝ := 50
  let cream_butterfat : ButterfatPercentage := ⟨23, by norm_num⟩
  let final_butterfat : ButterfatPercentage := ⟨3, by norm_num⟩
  let result := initial_butterfat_percentage initial_volume cream_volume cream_butterfat final_butterfat
  result.val = 4 := by sorry

end NUMINAMATH_CALUDE_milk_butterfat_calculation_l3643_364326


namespace NUMINAMATH_CALUDE_square_perimeter_l3643_364353

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (5 * s / 2 = 40) → (4 * s = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3643_364353


namespace NUMINAMATH_CALUDE_fourth_root_squared_l3643_364386

theorem fourth_root_squared (y : ℝ) : (y^(1/4))^2 = 81 → y = 81 := by sorry

end NUMINAMATH_CALUDE_fourth_root_squared_l3643_364386


namespace NUMINAMATH_CALUDE_newspaper_delivery_ratio_l3643_364372

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- The number of newspapers Jake delivers in a week -/
def jake_weekly : ℕ := 234

/-- The additional number of newspapers Miranda delivers compared to Jake in a month -/
def miranda_monthly_extra : ℕ := 936

/-- The ratio of newspapers Miranda delivers to Jake's deliveries in a week -/
def delivery_ratio : ℚ := (jake_weekly * weeks_per_month + miranda_monthly_extra) / (jake_weekly * weeks_per_month)

theorem newspaper_delivery_ratio :
  delivery_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_newspaper_delivery_ratio_l3643_364372


namespace NUMINAMATH_CALUDE_abc_value_l3643_364341

theorem abc_value (a b c : ℝ) 
  (h1 : a^2 * (b + c) = 2011)
  (h2 : b^2 * (a + c) = 2011)
  (h3 : a ≠ b) : 
  a * b * c = -2011 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l3643_364341


namespace NUMINAMATH_CALUDE_nap_start_time_l3643_364342

def minutes_past_midnight (hours minutes : ℕ) : ℕ :=
  hours * 60 + minutes

def time_from_minutes (total_minutes : ℕ) : ℕ × ℕ :=
  (total_minutes / 60, total_minutes % 60)

theorem nap_start_time 
  (nap_duration : ℕ) 
  (wake_up_hours wake_up_minutes : ℕ) 
  (h1 : nap_duration = 65)
  (h2 : wake_up_hours = 13)
  (h3 : wake_up_minutes = 30) :
  time_from_minutes (minutes_past_midnight wake_up_hours wake_up_minutes - nap_duration) = (12, 25) := by
  sorry

end NUMINAMATH_CALUDE_nap_start_time_l3643_364342


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3643_364384

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1365 → 
  L = 1575 → 
  L = 7 * S + R → 
  R = 105 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3643_364384


namespace NUMINAMATH_CALUDE_remainder_problem_l3643_364329

theorem remainder_problem (n : ℤ) (h : n % 7 = 3) : (4 * n - 9) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3643_364329


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l3643_364361

/-- Given an election with a total of 5200 votes where the winning candidate
    has a majority of 1040 votes, prove that the winning candidate received 60% of the votes. -/
theorem winning_candidate_percentage (total_votes : ℕ) (majority : ℕ) 
  (h_total : total_votes = 5200)
  (h_majority : majority = 1040) :
  (majority : ℚ) / total_votes * 100 + 50 = 60 := by
  sorry

end NUMINAMATH_CALUDE_winning_candidate_percentage_l3643_364361


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3643_364350

theorem quadratic_inequality_empty_solution_set (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) → k ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3643_364350


namespace NUMINAMATH_CALUDE_greatest_q_plus_r_l3643_364344

theorem greatest_q_plus_r : ∃ (q r : ℕ+), 
  927 = 21 * q + r ∧ 
  ∀ (q' r' : ℕ+), 927 = 21 * q' + r' → q + r ≥ q' + r' :=
by sorry

end NUMINAMATH_CALUDE_greatest_q_plus_r_l3643_364344


namespace NUMINAMATH_CALUDE_square_minus_product_identity_l3643_364394

theorem square_minus_product_identity (x y : ℝ) :
  (2*x - 3*y)^2 - (2*x + 3*y)*(2*x - 3*y) = -12*x*y + 18*y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_identity_l3643_364394


namespace NUMINAMATH_CALUDE_y_in_terms_of_z_l3643_364391

theorem y_in_terms_of_z (x y z : ℝ) : 
  x = 90 * (1 + 0.11) →
  y = x * (1 - 0.27) →
  z = y/2 + 3 →
  y = 2*z - 6 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_z_l3643_364391


namespace NUMINAMATH_CALUDE_product_expansion_l3643_364320

theorem product_expansion (x : ℝ) : 
  (x^2 + 3*x - 4) * (2*x^2 - x + 5) = 2*x^4 + 5*x^3 - 6*x^2 + 19*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3643_364320


namespace NUMINAMATH_CALUDE_percentage_sum_l3643_364318

theorem percentage_sum (P Q R x y : ℝ) 
  (h_pos_P : P > 0) (h_pos_Q : Q > 0) (h_pos_R : R > 0)
  (h_PQ : P = (1 + x / 100) * Q)
  (h_QR : Q = (1 + y / 100) * R)
  (h_PR : P = 2.4 * R) : 
  x + y = 140 := by sorry

end NUMINAMATH_CALUDE_percentage_sum_l3643_364318


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_l3643_364340

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a2 (a : ℕ → ℚ) 
  (h1 : geometric_sequence a) 
  (h2 : a 1 + a 3 = 4) : 
  a 2 = 8/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_l3643_364340


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3643_364352

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |a * x - 1|

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Theorem statement
theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, is_even (f a) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3643_364352


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_range_of_b_plus_c_l3643_364380

/-- Triangle ABC with sides a, b, c corresponding to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle --/
def triangle_condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.sin t.C + t.a * Real.cos t.C = t.c + t.b

/-- Theorem 1: Angle A is 60° --/
theorem angle_A_is_60_degrees (t : Triangle) (h : triangle_condition t) : t.A = π / 3 := by
  sorry

/-- Theorem 2: Range of b + c when a = √3 --/
theorem range_of_b_plus_c (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = Real.sqrt 3) :
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_range_of_b_plus_c_l3643_364380


namespace NUMINAMATH_CALUDE_colored_plane_congruent_triangle_l3643_364317

/-- A color type representing the 1992 colors -/
inductive Color
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10
-- ... (omitted for brevity, but in reality, this would list all 1992 colors)
| c1992

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle on the plane -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- A colored plane -/
def ColoredPlane := Point → Color

/-- Two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- A point is an interior point of a line segment -/
def isInteriorPoint (p : Point) (a b : Point) : Prop := sorry

/-- The theorem to be proved -/
theorem colored_plane_congruent_triangle 
  (plane : ColoredPlane) (T : Triangle) : 
  ∃ T' : Triangle, congruent T T' ∧ 
    (∀ (p q : Point), 
      ((isInteriorPoint p T'.a T'.b ∧ isInteriorPoint q T'.b T'.c) ∨
       (isInteriorPoint p T'.b T'.c ∧ isInteriorPoint q T'.c T'.a) ∨
       (isInteriorPoint p T'.c T'.a ∧ isInteriorPoint q T'.a T'.b)) →
      plane p = plane q) :=
sorry

end NUMINAMATH_CALUDE_colored_plane_congruent_triangle_l3643_364317


namespace NUMINAMATH_CALUDE_total_books_l3643_364359

theorem total_books (joan_books tom_books sarah_books alex_books : ℕ) 
  (h1 : joan_books = 10)
  (h2 : tom_books = 38)
  (h3 : sarah_books = 25)
  (h4 : alex_books = 45) :
  joan_books + tom_books + sarah_books + alex_books = 118 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l3643_364359


namespace NUMINAMATH_CALUDE_power_one_third_five_l3643_364354

theorem power_one_third_five : (1/3 : ℚ)^5 = 1/243 := by sorry

end NUMINAMATH_CALUDE_power_one_third_five_l3643_364354


namespace NUMINAMATH_CALUDE_greater_than_implies_half_greater_than_l3643_364346

theorem greater_than_implies_half_greater_than (a b : ℝ) (h : a > b) : a / 2 > b / 2 := by
  sorry

end NUMINAMATH_CALUDE_greater_than_implies_half_greater_than_l3643_364346


namespace NUMINAMATH_CALUDE_power_of_power_negative_l3643_364347

theorem power_of_power_negative (a : ℝ) : -(a^3)^4 = -a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_negative_l3643_364347


namespace NUMINAMATH_CALUDE_rope_purchase_difference_l3643_364303

def inches_per_foot : ℕ := 12

def last_week_purchase : ℕ := 6

def this_week_purchase_inches : ℕ := 96

theorem rope_purchase_difference :
  last_week_purchase - (this_week_purchase_inches / inches_per_foot) = 2 :=
by sorry

end NUMINAMATH_CALUDE_rope_purchase_difference_l3643_364303


namespace NUMINAMATH_CALUDE_no_valid_m_exists_l3643_364337

theorem no_valid_m_exists : ¬ ∃ (m : ℝ),
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 1 2 →
    x₁ + m > x₂^2 - m*x₂ + m^2/2 + 2*m - 3) ∧
  (Set.Ioo 1 2 = {x | x^2 - m*x + m^2/2 + 2*m - 3 < m^2/2 + 1}) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_m_exists_l3643_364337


namespace NUMINAMATH_CALUDE_partial_fraction_coefficient_sum_l3643_364304

theorem partial_fraction_coefficient_sum :
  ∀ (A B C D E : ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_coefficient_sum_l3643_364304


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3643_364330

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3643_364330


namespace NUMINAMATH_CALUDE_base_2_representation_of_125_l3643_364365

theorem base_2_representation_of_125 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 0 ∧ g = 1) ∧
    125 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_125_l3643_364365


namespace NUMINAMATH_CALUDE_school_problem_solution_l3643_364382

/-- Represents the number of students in each class of a school -/
structure School where
  class1 : ℕ
  class2 : ℕ
  class3 : ℕ
  class4 : ℕ
  class5 : ℕ

/-- The conditions of the school problem -/
def SchoolProblem (s : School) : Prop :=
  s.class1 = 23 ∧
  s.class2 < s.class1 ∧
  s.class3 < s.class2 ∧
  s.class4 < s.class3 ∧
  s.class5 < s.class4 ∧
  s.class1 + s.class2 + s.class3 + s.class4 + s.class5 = 95 ∧
  ∃ (x : ℕ), 
    s.class2 = s.class1 - x ∧
    s.class3 = s.class2 - x ∧
    s.class4 = s.class3 - x ∧
    s.class5 = s.class4 - x

theorem school_problem_solution (s : School) (h : SchoolProblem s) :
  ∃ (x : ℕ), x = 2 ∧
    s.class2 = s.class1 - x ∧
    s.class3 = s.class2 - x ∧
    s.class4 = s.class3 - x ∧
    s.class5 = s.class4 - x :=
  sorry

end NUMINAMATH_CALUDE_school_problem_solution_l3643_364382


namespace NUMINAMATH_CALUDE_data_plan_total_cost_l3643_364374

/-- Calculates the total cost of a data plan over 6 months with special conditions -/
def data_plan_cost (regular_charge : ℚ) (promo_rate : ℚ) (extra_fee : ℚ) : ℚ :=
  let first_month := regular_charge * promo_rate
  let fourth_month := regular_charge + extra_fee
  let regular_months := 4 * regular_charge
  first_month + fourth_month + regular_months

/-- Proves that the total cost for the given conditions is $175 -/
theorem data_plan_total_cost :
  data_plan_cost 30 (1/3) 15 = 175 := by
  sorry

#eval data_plan_cost 30 (1/3) 15

end NUMINAMATH_CALUDE_data_plan_total_cost_l3643_364374


namespace NUMINAMATH_CALUDE_basketball_league_games_l3643_364300

/-- The number of games played in a basketball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264. -/
theorem basketball_league_games :
  total_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l3643_364300


namespace NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l3643_364305

/-- The number of products inspected --/
def n : ℕ := 10

/-- Event A: at least two defective products --/
def event_A (x : ℕ) : Prop := x ≥ 2

/-- The complementary event of A --/
def complement_A (x : ℕ) : Prop := x ≤ 1

/-- Theorem stating that the complement of "at least two defective products" 
    is "at most one defective product" --/
theorem complement_of_at_least_two_defective :
  ∀ x : ℕ, x ≤ n → (¬ event_A x ↔ complement_A x) := by sorry

end NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l3643_364305


namespace NUMINAMATH_CALUDE_wine_sales_regression_l3643_364366

/-- Linear regression problem for white wine sales and unit cost -/
theorem wine_sales_regression 
  (x_mean : ℝ) 
  (y_mean : ℝ) 
  (sum_x_squared : ℝ) 
  (sum_xy : ℝ) 
  (n : ℕ) 
  (h_x_mean : x_mean = 7/2)
  (h_y_mean : y_mean = 71)
  (h_sum_x_squared : sum_x_squared = 79)
  (h_sum_xy : sum_xy = 1481)
  (h_n : n = 6) :
  let b := (sum_xy - n * x_mean * y_mean) / (sum_x_squared - n * x_mean^2)
  ∃ ε > 0, |b + 1.8182| < ε :=
sorry

end NUMINAMATH_CALUDE_wine_sales_regression_l3643_364366


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3643_364396

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 14*m + 40 ≤ 0 → n ≤ m) ∧ n^2 - 14*n + 40 ≤ 0 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3643_364396


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l3643_364333

theorem min_value_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x - 2*y + 3*z = 0) :
  y^2 / (x*z) ≥ 3 := by
  sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x - 2*y + 3*z = 0 ∧ y^2 / (x*z) < 3 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l3643_364333


namespace NUMINAMATH_CALUDE_correct_cube_root_l3643_364355

-- Define the cube root function for real numbers
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Theorem statement
theorem correct_cube_root : cubeRoot (-125) = -5 := by
  sorry

end NUMINAMATH_CALUDE_correct_cube_root_l3643_364355


namespace NUMINAMATH_CALUDE_winning_strategy_l3643_364362

/-- Represents the player who has a winning strategy -/
inductive WinningPlayer
  | First
  | Second

/-- Defines the game on a grid board -/
def gridGame (m n : ℕ) : WinningPlayer :=
  if m % 2 = 0 ∧ n % 2 = 0 then
    WinningPlayer.Second
  else if m % 2 = 1 ∧ n % 2 = 1 then
    WinningPlayer.Second
  else
    WinningPlayer.First

/-- Theorem stating the winning strategy for different board sizes -/
theorem winning_strategy :
  (gridGame 10 12 = WinningPlayer.Second) ∧
  (gridGame 9 10 = WinningPlayer.First) ∧
  (gridGame 9 11 = WinningPlayer.Second) :=
sorry

end NUMINAMATH_CALUDE_winning_strategy_l3643_364362


namespace NUMINAMATH_CALUDE_first_day_of_month_is_sunday_l3643_364307

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the day of the week for a given day of the month
def dayOfWeek (dayOfMonth : Nat) : DayOfWeek := sorry

-- Theorem statement
theorem first_day_of_month_is_sunday 
  (h : dayOfWeek 18 = DayOfWeek.Wednesday) : 
  dayOfWeek 1 = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_first_day_of_month_is_sunday_l3643_364307


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l3643_364358

theorem investment_interest_calculation (total_investment : ℝ) (first_investment : ℝ) 
  (first_rate : ℝ) (second_rate : ℝ) (h1 : total_investment = 10000) 
  (h2 : first_investment = 6000) (h3 : first_rate = 0.09) (h4 : second_rate = 0.11) : 
  first_investment * first_rate + (total_investment - first_investment) * second_rate = 980 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l3643_364358


namespace NUMINAMATH_CALUDE_doug_money_l3643_364325

/-- Represents the amount of money each person has -/
structure Money where
  josh : ℚ
  doug : ℚ
  brad : ℚ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.josh + m.doug + m.brad = 68 ∧
  m.josh = 2 * m.brad ∧
  m.josh = 3/4 * m.doug

/-- The theorem to prove -/
theorem doug_money (m : Money) (h : problem_conditions m) : m.doug = 32 := by
  sorry

end NUMINAMATH_CALUDE_doug_money_l3643_364325


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l3643_364357

theorem least_three_digit_multiple_of_13 : ∃ (n : ℕ), 
  n = 104 ∧ 
  13 ∣ n ∧ 
  100 ≤ n ∧ 
  n < 1000 ∧
  ∀ m : ℕ, (13 ∣ m ∧ 100 ≤ m ∧ m < 1000) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l3643_364357


namespace NUMINAMATH_CALUDE_unique_congruent_integer_l3643_364393

theorem unique_congruent_integer (h : ∃ m : ℤ, 10 ≤ m ∧ m ≤ 15 ∧ m ≡ 9433 [ZMOD 7]) :
  ∃! m : ℤ, 10 ≤ m ∧ m ≤ 15 ∧ m ≡ 9433 [ZMOD 7] ∧ m = 14 :=
by sorry

end NUMINAMATH_CALUDE_unique_congruent_integer_l3643_364393


namespace NUMINAMATH_CALUDE_room_width_proof_l3643_364313

/-- Given a rectangular room with specified dimensions and veranda, prove its width. -/
theorem room_width_proof (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) : 
  room_length = 17 →
  veranda_width = 2 →
  veranda_area = 132 →
  ∃ room_width : ℝ,
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - 
    (room_length * room_width) = veranda_area ∧
    room_width = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_width_proof_l3643_364313


namespace NUMINAMATH_CALUDE_square_of_five_equals_twentyfive_l3643_364328

theorem square_of_five_equals_twentyfive : (5 : ℕ)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_five_equals_twentyfive_l3643_364328


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_squared_l3643_364308

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  b : ℝ  -- length of the shorter base
  h : ℝ  -- height of the trapezoid
  midline_ratio : (b + 75) / (b + 25) = 3 / 2  -- ratio condition for the midline
  x : ℝ  -- length of the segment dividing the trapezoid into two equal areas
  equal_area_condition : x = 125 * (100 / (x - 75)) - 75

/-- The main theorem about the trapezoid -/
theorem trapezoid_segment_length_squared (t : Trapezoid) :
  ⌊(t.x^2) / 100⌋ = 181 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_squared_l3643_364308


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l3643_364309

theorem max_digits_product_5_4 : 
  ∀ (a b : ℕ), 
    10000 ≤ a ∧ a < 100000 →
    1000 ≤ b ∧ b < 10000 →
    a * b < 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l3643_364309


namespace NUMINAMATH_CALUDE_max_puns_purchase_l3643_364378

/-- Represents the cost of each item --/
structure ItemCosts where
  pin : ℕ
  pon : ℕ
  pun : ℕ

/-- Represents the quantity of each item purchased --/
structure Purchase where
  pins : ℕ
  pons : ℕ
  puns : ℕ

/-- Calculates the total cost of a purchase --/
def totalCost (costs : ItemCosts) (purchase : Purchase) : ℕ :=
  costs.pin * purchase.pins + costs.pon * purchase.pons + costs.pun * purchase.puns

/-- Checks if a purchase is valid (at least one of each item) --/
def isValidPurchase (purchase : Purchase) : Prop :=
  purchase.pins ≥ 1 ∧ purchase.pons ≥ 1 ∧ purchase.puns ≥ 1

/-- The main theorem statement --/
theorem max_puns_purchase (costs : ItemCosts) (budget : ℕ) : 
  costs.pin = 3 → costs.pon = 4 → costs.pun = 9 → budget = 108 →
  ∃ (max_puns : ℕ), 
    (∃ (p : Purchase), isValidPurchase p ∧ totalCost costs p = budget ∧ p.puns = max_puns) ∧
    (∀ (p : Purchase), isValidPurchase p → totalCost costs p = budget → p.puns ≤ max_puns) ∧
    max_puns = 10 :=
sorry

end NUMINAMATH_CALUDE_max_puns_purchase_l3643_364378


namespace NUMINAMATH_CALUDE_arrangement_count_is_2028_l3643_364368

/-- Represents the set of files that can be arranged after lunch -/
def RemainingFiles : Finset ℕ := Finset.range 9 ∪ {12}

/-- The number of ways to arrange a subset of files from {1,2,...,9,12} -/
def ArrangementCount : ℕ := sorry

/-- Theorem stating that the number of different arrangements is 2028 -/
theorem arrangement_count_is_2028 : ArrangementCount = 2028 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_2028_l3643_364368


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l3643_364370

theorem maximum_marks_calculation (passing_threshold : ℝ) (scored_marks : ℕ) (shortfall : ℕ) : 
  passing_threshold = 30 / 100 →
  scored_marks = 212 →
  shortfall = 16 →
  ∃ (total_marks : ℕ), total_marks = 760 ∧ 
    (scored_marks + shortfall : ℝ) / total_marks = passing_threshold :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l3643_364370


namespace NUMINAMATH_CALUDE_peter_train_probability_l3643_364339

theorem peter_train_probability (p : ℚ) (h : p = 5/12) : 1 - p = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_peter_train_probability_l3643_364339


namespace NUMINAMATH_CALUDE_scientific_notation_35000_l3643_364301

theorem scientific_notation_35000 :
  35000 = 3.5 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_35000_l3643_364301


namespace NUMINAMATH_CALUDE_game_time_calculation_l3643_364356

/-- Calculates the total time before playing a game given download, installation, and tutorial times. -/
def totalGameTime (downloadTime : ℕ) : ℕ :=
  let installTime := downloadTime / 2
  let combinedTime := downloadTime + installTime
  let tutorialTime := 3 * combinedTime
  combinedTime + tutorialTime

/-- Theorem stating that for a download time of 10 minutes, the total time before playing is 60 minutes. -/
theorem game_time_calculation :
  totalGameTime 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_game_time_calculation_l3643_364356


namespace NUMINAMATH_CALUDE_binomial_distribution_parameters_l3643_364397

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_parameters 
  (X : BinomialDistribution) 
  (h_expectation : expectation X = 2)
  (h_variance : variance X = 4) :
  X.n = 12 ∧ X.p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_parameters_l3643_364397


namespace NUMINAMATH_CALUDE_mashas_juice_theorem_l3643_364383

/-- Represents Masha's juice drinking process over 3 days -/
def mashas_juice_process (x : ℝ) : Prop :=
  let day1_juice := x - 1
  let day2_juice := (day1_juice^2) / x
  let day3_juice := (day2_juice^2) / x
  let final_juice := (day3_juice^2) / x
  let final_water := x - final_juice
  (final_water = final_juice + 1.5) ∧ (x > 1)

/-- The theorem stating the result of Masha's juice drinking process -/
theorem mashas_juice_theorem :
  ∀ x : ℝ, mashas_juice_process x ↔ (x = 2 ∧ (2 - ((2 - 1)^3) / 2^2 = 1.75)) :=
by sorry

end NUMINAMATH_CALUDE_mashas_juice_theorem_l3643_364383


namespace NUMINAMATH_CALUDE_max_pencils_buyable_l3643_364373

def total_money : ℚ := 36
def pencil_cost : ℚ := 1.80
def pen_cost : ℚ := 2.60
def num_pens : ℕ := 9

theorem max_pencils_buyable :
  ∃ (num_pencils : ℕ),
    (num_pencils * pencil_cost + num_pens * pen_cost ≤ total_money) ∧
    ((num_pencils + num_pens) % 3 = 0) ∧
    (∀ (n : ℕ), n > num_pencils →
      (n * pencil_cost + num_pens * pen_cost > total_money ∨
       (n + num_pens) % 3 ≠ 0)) ∧
    num_pencils = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_pencils_buyable_l3643_364373


namespace NUMINAMATH_CALUDE_pear_sales_l3643_364336

theorem pear_sales (morning_sales afternoon_sales total_sales : ℕ) :
  morning_sales = 120 →
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 360 →
  afternoon_sales = 240 := by
sorry

end NUMINAMATH_CALUDE_pear_sales_l3643_364336


namespace NUMINAMATH_CALUDE_vector_angle_problem_l3643_364343

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_angle_problem (b : ℝ × ℝ) :
  let a : ℝ × ℝ := (1, Real.sqrt 3)
  (Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) →
  (Real.sqrt (((a.1 + b.1) ^ 2) + ((a.2 + b.2) ^ 2)) = Real.sqrt 3) →
  angle_between_vectors a b = (2 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l3643_364343


namespace NUMINAMATH_CALUDE_cube_root_of_product_l3643_364319

theorem cube_root_of_product (a b c : ℕ) : 
  (2^9 * 3^6 * 7^3 : ℝ)^(1/3) = 504 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l3643_364319


namespace NUMINAMATH_CALUDE_magnitude_of_p_l3643_364314

def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem magnitude_of_p (a b p : ℝ × ℝ) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (hab : a.1 * b.1 + a.2 * b.2 = -1/2) 
  (hpa : p.1 * a.1 + p.2 * a.2 = 1/2) 
  (hpb : p.1 * b.1 + p.2 * b.2 = 1/2) : 
  p.1^2 + p.2^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_p_l3643_364314


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l3643_364323

theorem roots_sum_and_product (p q : ℝ) : 
  p^2 - 5*p + 7 = 0 → 
  q^2 - 5*q + 7 = 0 → 
  p^3 + p^4*q^2 + p^2*q^4 + q^3 = 559 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l3643_364323


namespace NUMINAMATH_CALUDE_power_sum_negatives_l3643_364364

theorem power_sum_negatives (n : ℕ) : (-2)^n + (-2)^(n+1) = 2^n := by sorry

end NUMINAMATH_CALUDE_power_sum_negatives_l3643_364364


namespace NUMINAMATH_CALUDE_triangle_inequality_l3643_364351

/-- Given a triangle ABC with circumradius R, inradius r, and semiperimeter p,
    prove that 16 R r - 5 r^2 ≤ p^2 ≤ 4 R^2 + 4 R r + 3 r^2 --/
theorem triangle_inequality (R r p : ℝ) (hR : R > 0) (hr : r > 0) (hp : p > 0) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3643_364351


namespace NUMINAMATH_CALUDE_problem_solution_l3643_364302

theorem problem_solution :
  (∀ x : ℝ, x^2 + x + 2 ≥ 0) ∧
  (∀ x y : ℝ, x * y = ((x + y) / 2)^2 ↔ x = y) ∧
  (∃ p q : Prop, ¬(p ∧ q) ∧ ¬(¬p ∧ ¬q)) ∧
  (∀ A B C : ℝ, ∀ sinA sinB : ℝ, 
    sinA = Real.sin A ∧ sinB = Real.sin B →
    sinA > sinB → A > B) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3643_364302


namespace NUMINAMATH_CALUDE_euler_conjecture_counterexample_l3643_364363

theorem euler_conjecture_counterexample : 133^5 + 110^5 + 84^5 + 27^5 = 144^5 := by
  sorry

end NUMINAMATH_CALUDE_euler_conjecture_counterexample_l3643_364363


namespace NUMINAMATH_CALUDE_hydrangea_year_calculation_l3643_364367

/-- The year Lily started buying hydrangeas -/
def start_year : ℕ := 1989

/-- The cost of each hydrangea plant in dollars -/
def plant_cost : ℕ := 20

/-- The total amount Lily has spent on hydrangeas in dollars -/
def total_spent : ℕ := 640

/-- The year up to which Lily has spent the total amount on hydrangeas -/
def end_year : ℕ := 2021

/-- Theorem stating that the calculated end year is correct -/
theorem hydrangea_year_calculation :
  end_year = start_year + (total_spent / plant_cost) :=
by sorry

end NUMINAMATH_CALUDE_hydrangea_year_calculation_l3643_364367


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l3643_364338

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/8]
  (∀ x ∈ sums, x ≤ 1/3 + 1/2) ∧ (1/3 + 1/2 = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l3643_364338


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l3643_364392

/-- Probability of a palindrome in a four-letter sequence -/
def prob_letter_palindrome : ℚ := 1 / 676

/-- Probability of a palindrome in a four-digit sequence -/
def prob_digit_palindrome : ℚ := 1 / 100

/-- Total number of possible license plate arrangements -/
def total_arrangements : ℕ := 26^4 * 10^4

theorem license_plate_palindrome_probability :
  let prob_at_least_one_palindrome := prob_letter_palindrome + prob_digit_palindrome - 
                                      (prob_letter_palindrome * prob_digit_palindrome)
  prob_at_least_one_palindrome = 775 / 67600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l3643_364392


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l3643_364398

theorem quadratic_form_equivalence :
  ∀ x y : ℝ, y = (1/2) * x^2 - 2*x + 1 ↔ y = (1/2) * (x - 2)^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l3643_364398


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l3643_364381

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (6 * x) % 31 = 19 % 31 ∧ 
  ∀ (y : ℕ), y > 0 → (6 * y) % 31 = 19 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l3643_364381


namespace NUMINAMATH_CALUDE_hcl_moles_formed_l3643_364311

-- Define the chemical equation
structure ChemicalEquation where
  reactants : List (String × ℕ)
  products : List (String × ℕ)

-- Define the reaction
def reaction : ChemicalEquation :=
  { reactants := [("CH4", 1), ("Cl2", 4)],
    products := [("CCl4", 1), ("HCl", 4)] }

-- Define the initial quantities
def initialQuantities : List (String × ℕ) :=
  [("CH4", 1), ("Cl2", 4)]

-- Theorem to prove
theorem hcl_moles_formed (reaction : ChemicalEquation) (initialQuantities : List (String × ℕ)) :
  reaction.reactants = [("CH4", 1), ("Cl2", 4)] →
  reaction.products = [("CCl4", 1), ("HCl", 4)] →
  initialQuantities = [("CH4", 1), ("Cl2", 4)] →
  (List.find? (λ p => p.1 = "HCl") reaction.products).map Prod.snd = some 4 := by
  sorry

end NUMINAMATH_CALUDE_hcl_moles_formed_l3643_364311


namespace NUMINAMATH_CALUDE_quadratic_one_solution_find_m_l3643_364331

/-- A quadratic equation ax² + bx + c = 0 has exactly one solution if and only if its discriminant b² - 4ac = 0 -/
theorem quadratic_one_solution (a b c : ℝ) (h : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = 0) ↔ b^2 - 4*a*c = 0 := by sorry

theorem find_m : ∃ m : ℚ, (∃! x : ℝ, 3 * x^2 - 7 * x + m = 0) → m = 49/12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_find_m_l3643_364331


namespace NUMINAMATH_CALUDE_angle_B_is_45_degrees_l3643_364395

theorem angle_B_is_45_degrees (A B : ℝ) 
  (h : 90 - (A + B) = 180 - (A - B)) : B = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_45_degrees_l3643_364395


namespace NUMINAMATH_CALUDE_num_divisors_360_eq_24_l3643_364334

/-- The number of positive divisors of 360 -/
def num_divisors_360 : ℕ := sorry

/-- Theorem stating that the number of positive divisors of 360 is 24 -/
theorem num_divisors_360_eq_24 : num_divisors_360 = 24 := by sorry

end NUMINAMATH_CALUDE_num_divisors_360_eq_24_l3643_364334


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3643_364369

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x^2 > 4}

-- Define set N
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- The theorem to prove
theorem intersection_complement_theorem :
  N ∩ (Set.compl M) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3643_364369


namespace NUMINAMATH_CALUDE_future_sales_prediction_l3643_364399

def current_year_sales : ℕ := 327
def yearly_increase : ℕ := 50
def years_ahead : ℕ := 3

theorem future_sales_prediction :
  current_year_sales + yearly_increase * years_ahead = 477 :=
by sorry

end NUMINAMATH_CALUDE_future_sales_prediction_l3643_364399


namespace NUMINAMATH_CALUDE_C_not_necessarily_necessary_for_A_C_not_necessarily_sufficient_for_A_l3643_364322

-- Define propositions A, B, and C
variable (A B C : Prop)

-- C is a necessary condition for B
axiom C_necessary_for_B : B → C

-- B is a sufficient condition for A
axiom B_sufficient_for_A : B → A

-- Theorem: C is not necessarily a necessary condition for A
theorem C_not_necessarily_necessary_for_A : ¬(A → C) := by sorry

-- Theorem: C is not necessarily a sufficient condition for A
theorem C_not_necessarily_sufficient_for_A : ¬(C → A) := by sorry

end NUMINAMATH_CALUDE_C_not_necessarily_necessary_for_A_C_not_necessarily_sufficient_for_A_l3643_364322


namespace NUMINAMATH_CALUDE_f_min_f_min_range_g_max_min_l3643_364324

-- Define the function f(x) = |x-2| + |x-3|
def f (x : ℝ) : ℝ := |x - 2| + |x - 3|

-- Define the function g(x) = |x-2| + |x-3| - |x-1|
def g (x : ℝ) : ℝ := |x - 2| + |x - 3| - |x - 1|

-- Theorem stating the minimum value of f(x)
theorem f_min : ∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y ∧ f x = 1 :=
sorry

-- Theorem stating the range where f(x) is minimized
theorem f_min_range : ∀ (x : ℝ), f x = 1 → 2 ≤ x ∧ x < 3 :=
sorry

-- Main theorem
theorem g_max_min :
  (∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y) →
  (∃ (a b : ℝ), (∀ (x : ℝ), f x = 1 → g x ≤ a ∧ b ≤ g x) ∧ a = 0 ∧ b = -1) :=
sorry

end NUMINAMATH_CALUDE_f_min_f_min_range_g_max_min_l3643_364324


namespace NUMINAMATH_CALUDE_largest_power_l3643_364335

theorem largest_power (a b c d e : ℕ) :
  a = 1 ∧ b = 2 ∧ c = 4 ∧ d = 8 ∧ e = 16 →
  c^8 ≥ a^20 ∧ c^8 ≥ b^14 ∧ c^8 ≥ d^5 ∧ c^8 ≥ e^3 :=
by sorry

#check largest_power

end NUMINAMATH_CALUDE_largest_power_l3643_364335


namespace NUMINAMATH_CALUDE_problem_solution_l3643_364385

def A : Set ℝ := {x | |x - 2| < 3}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

theorem problem_solution :
  (∀ x, x ∈ (A ∩ (Set.univ \ B 3)) ↔ 3 ≤ x ∧ x < 5) ∧
  (A ∩ B 8 = {x | -1 < x ∧ x < 4}) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3643_364385


namespace NUMINAMATH_CALUDE_gwen_race_results_l3643_364316

/-- Represents the race details --/
structure RaceDetails where
  jogging_time : ℕ
  jogging_elevation : ℕ
  jogging_ratio : ℕ
  walking_ratio : ℕ

/-- Calculates the walking time based on race details --/
def walking_time (race : RaceDetails) : ℕ :=
  (race.jogging_time / race.jogging_ratio) * race.walking_ratio

/-- Calculates the total elevation gain based on race details --/
def total_elevation_gain (race : RaceDetails) : ℕ :=
  (race.jogging_elevation * (race.jogging_time + walking_time race)) / race.jogging_time

/-- Theorem stating the walking time and total elevation gain for Gwen's race --/
theorem gwen_race_results (race : RaceDetails) 
  (h1 : race.jogging_time = 15)
  (h2 : race.jogging_elevation = 500)
  (h3 : race.jogging_ratio = 5)
  (h4 : race.walking_ratio = 3) :
  walking_time race = 9 ∧ total_elevation_gain race = 800 := by
  sorry


end NUMINAMATH_CALUDE_gwen_race_results_l3643_364316


namespace NUMINAMATH_CALUDE_game_ends_after_28_rounds_l3643_364327

/-- Represents the state of the game at any given round -/
structure GameState where
  x : Nat
  y : Nat
  z : Nat

/-- Represents the rules of the token redistribution game -/
def redistributeTokens (state : GameState) : GameState :=
  sorry

/-- Determines if the game has ended (i.e., if any player has run out of tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def countRounds (state : GameState) : Nat :=
  sorry

/-- Theorem stating that the game ends after 28 rounds -/
theorem game_ends_after_28_rounds :
  countRounds (GameState.mk 18 15 12) = 28 := by
  sorry

end NUMINAMATH_CALUDE_game_ends_after_28_rounds_l3643_364327


namespace NUMINAMATH_CALUDE_wang_hao_height_l3643_364349

/-- Given Yao Ming's height and the difference between Yao Ming's and Wang Hao's heights,
    prove that Wang Hao's height is 1.58 meters. -/
theorem wang_hao_height (yao_ming_height : ℝ) (height_difference : ℝ) 
  (h1 : yao_ming_height = 2.29)
  (h2 : height_difference = 0.71) :
  yao_ming_height - height_difference = 1.58 := by
  sorry

end NUMINAMATH_CALUDE_wang_hao_height_l3643_364349


namespace NUMINAMATH_CALUDE_brent_initial_lollipops_l3643_364312

/-- The number of lollipops Brent initially received -/
def initial_lollipops : ℕ := sorry

/-- The number of Kit-Kat bars Brent received -/
def kit_kat : ℕ := 5

/-- The number of Hershey kisses Brent received -/
def hershey_kisses : ℕ := 3 * kit_kat

/-- The number of boxes of Nerds Brent received -/
def nerds : ℕ := 8

/-- The number of Baby Ruths Brent had -/
def baby_ruths : ℕ := 10

/-- The number of Reese's Peanut Butter Cups Brent had -/
def reeses_cups : ℕ := baby_ruths / 2

/-- The number of lollipops Brent gave to his sister -/
def lollipops_given : ℕ := 5

/-- The total number of candy pieces Brent had after giving away lollipops -/
def remaining_candy : ℕ := 49

theorem brent_initial_lollipops :
  initial_lollipops = 11 :=
by sorry

end NUMINAMATH_CALUDE_brent_initial_lollipops_l3643_364312
