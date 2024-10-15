import Mathlib

namespace NUMINAMATH_CALUDE_fewer_bees_than_flowers_l1111_111162

theorem fewer_bees_than_flowers (flowers : ℕ) (bees : ℕ) 
  (h1 : flowers = 5) (h2 : bees = 3) : flowers - bees = 2 := by
  sorry

end NUMINAMATH_CALUDE_fewer_bees_than_flowers_l1111_111162


namespace NUMINAMATH_CALUDE_bills_initial_money_l1111_111161

def total_initial_money : ℕ := 42
def num_pizzas : ℕ := 3
def pizza_cost : ℕ := 11
def bill_final_money : ℕ := 39

theorem bills_initial_money :
  let frank_spent := num_pizzas * pizza_cost
  let frank_leftover := total_initial_money - frank_spent
  let bill_initial := bill_final_money - frank_leftover
  bill_initial = 30 := by
sorry

end NUMINAMATH_CALUDE_bills_initial_money_l1111_111161


namespace NUMINAMATH_CALUDE_ellipse_properties_l1111_111158

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Properties of a specific ellipse -/
theorem ellipse_properties (C : Ellipse) (P : Point) :
  P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1 →  -- P is on the ellipse
  P.x = 1 →                           -- P's x-coordinate is 1
  P.y = Real.sqrt 2 / 2 →             -- P's y-coordinate is √2/2
  (∃ F₁ F₂ : Point, |P.x - F₁.x| + |P.y - F₁.y| + |P.x - F₂.x| + |P.y - F₂.y| = 2 * Real.sqrt 2) →  -- Distance sum to foci is 2√2
  (C.a^2 = 2 ∧ C.b^2 = 1) ∧           -- Standard equation of C is x²/2 + y² = 1
  (∃ (A B O : Point) (l : Set Point),
    O = ⟨0, 0⟩ ∧                      -- O is the origin
    F₂ ∈ l ∧ A ∈ l ∧ B ∈ l ∧          -- l passes through F₂, A, and B
    A.x^2 / C.a^2 + A.y^2 / C.b^2 = 1 ∧  -- A is on the ellipse
    B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1 ∧  -- B is on the ellipse
    (∀ A' B' : Point,
      A' ∈ l → B' ∈ l →
      A'.x^2 / C.a^2 + A'.y^2 / C.b^2 = 1 →
      B'.x^2 / C.a^2 + B'.y^2 / C.b^2 = 1 →
      abs ((A.x - O.x) * (B.y - O.y) - (B.x - O.x) * (A.y - O.y)) / 2 ≥
      abs ((A'.x - O.x) * (B'.y - O.y) - (B'.x - O.x) * (A'.y - O.y)) / 2) ∧
    abs ((A.x - O.x) * (B.y - O.y) - (B.x - O.x) * (A.y - O.y)) / 2 = Real.sqrt 2 / 2) -- Max area of AOB is √2/2
  := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1111_111158


namespace NUMINAMATH_CALUDE_average_sum_of_abs_diff_l1111_111199

def sum_of_abs_diff (perm : Fin 8 → Fin 8) : ℕ :=
  |perm 0 - perm 1| + |perm 2 - perm 3| + |perm 4 - perm 5| + |perm 6 - perm 7|

def all_permutations : Finset (Fin 8 → Fin 8) :=
  Finset.univ.filter (fun f => Function.Injective f)

theorem average_sum_of_abs_diff :
  (Finset.sum all_permutations sum_of_abs_diff) / all_permutations.card = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_sum_of_abs_diff_l1111_111199


namespace NUMINAMATH_CALUDE_congruence_mod_10_l1111_111182

theorem congruence_mod_10 : ∃ C : ℤ, (1 + C * (2^20 - 1)) % 10 = 2011 % 10 := by
  sorry

end NUMINAMATH_CALUDE_congruence_mod_10_l1111_111182


namespace NUMINAMATH_CALUDE_school_trip_photos_l1111_111148

theorem school_trip_photos (claire_photos : ℕ) (lisa_photos : ℕ) (robert_photos : ℕ) :
  claire_photos = 10 →
  lisa_photos = 3 * claire_photos →
  robert_photos = claire_photos + 20 →
  lisa_photos + robert_photos = 60 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_photos_l1111_111148


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1111_111186

/-- An arithmetic sequence with first term 2 and the property a_2 + a_4 = a_6 has common difference 2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 2) 
  (h_sum_property : a 2 + a 4 = a 6) :
  ∀ n : ℕ, a (n + 1) - a n = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1111_111186


namespace NUMINAMATH_CALUDE_johnny_emily_meeting_distance_l1111_111173

-- Define the total distance
def total_distance : ℝ := 60

-- Define walking rates
def matthew_rate : ℝ := 3
def johnny_rate : ℝ := 4
def emily_rate : ℝ := 5

-- Define the time difference between Matthew's start and Johnny/Emily's start
def time_diff : ℝ := 1

-- Define the function to calculate the distance Johnny walked
def johnny_distance (t : ℝ) : ℝ := johnny_rate * t

-- Theorem statement
theorem johnny_emily_meeting_distance :
  ∃ t : ℝ, t > 0 ∧ 
    matthew_rate * (t + time_diff) + johnny_distance t + emily_rate * t = total_distance ∧
    johnny_distance t = 19 := by
  sorry

end NUMINAMATH_CALUDE_johnny_emily_meeting_distance_l1111_111173


namespace NUMINAMATH_CALUDE_rectangle_area_l1111_111172

/-- Given a rectangle with width 10 meters, if its length is increased such that the new area is 4/3 times the original area and the new perimeter is 60 meters, then the original area of the rectangle is 150 square meters. -/
theorem rectangle_area (original_length : ℝ) : 
  let original_width : ℝ := 10
  let new_length : ℝ := (60 - 2 * original_width) / 2
  let new_area : ℝ := new_length * original_width
  let original_area : ℝ := original_length * original_width
  new_area = (4/3) * original_area → original_area = 150 := by
sorry


end NUMINAMATH_CALUDE_rectangle_area_l1111_111172


namespace NUMINAMATH_CALUDE_triangle_similarity_problem_l1111_111197

theorem triangle_similarity_problem (DC CB AD : ℝ) (h1 : DC = 13) (h2 : CB = 9) 
  (h3 : AD > 0) (h4 : (1/3) * AD + DC + CB = AD) : 
  ∃ FC : ℝ, FC = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_problem_l1111_111197


namespace NUMINAMATH_CALUDE_triangle_has_inside_altitude_l1111_111119

-- Define a triangle
def Triangle : Type := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

-- Define an altitude of a triangle
def Altitude (t : Triangle) : Type := ℝ × ℝ × ℝ × ℝ

-- Define what it means for an altitude to be inside a triangle
def IsInside (a : Altitude t) (t : Triangle) : Prop := sorry

-- Theorem statement
theorem triangle_has_inside_altitude (t : Triangle) : 
  ∃ (a : Altitude t), IsInside a t := sorry

end NUMINAMATH_CALUDE_triangle_has_inside_altitude_l1111_111119


namespace NUMINAMATH_CALUDE_set_operations_l1111_111141

open Set

def A : Set ℝ := {x | 2 * x - 8 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}

theorem set_operations :
  (A ∩ B = {x : ℝ | 0 < x ∧ x < 4}) ∧
  ((Aᶜ ∪ B) = {x : ℝ | 0 < x}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1111_111141


namespace NUMINAMATH_CALUDE_number_division_problem_l1111_111166

theorem number_division_problem : ∃ x : ℝ, (x / 5 = 70 + x / 6) ∧ x = 2100 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1111_111166


namespace NUMINAMATH_CALUDE_range_of_f_is_real_l1111_111125

-- Define the function f
def f (x : ℝ) : ℝ := -4 * x + 5

-- Theorem stating that the range of f is ℝ
theorem range_of_f_is_real : Set.range f = Set.univ :=
sorry

end NUMINAMATH_CALUDE_range_of_f_is_real_l1111_111125


namespace NUMINAMATH_CALUDE_value_of_expression_l1111_111198

theorem value_of_expression (x y : ℝ) 
  (h1 : x^2 + x*y = 3) 
  (h2 : x*y + y^2 = -2) : 
  2*x^2 - x*y - 3*y^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l1111_111198


namespace NUMINAMATH_CALUDE_travel_options_count_l1111_111139

/-- The number of train services from location A to location B -/
def num_train_services : ℕ := 3

/-- The number of ferry services from location B to location C -/
def num_ferry_services : ℕ := 2

/-- The total number of travel options from location A to location C -/
def total_travel_options : ℕ := num_train_services * num_ferry_services

theorem travel_options_count : total_travel_options = 6 := by
  sorry

end NUMINAMATH_CALUDE_travel_options_count_l1111_111139


namespace NUMINAMATH_CALUDE_xy_squared_minus_y_squared_x_equals_zero_l1111_111121

theorem xy_squared_minus_y_squared_x_equals_zero (x y : ℝ) : x * y^2 - y^2 * x = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_minus_y_squared_x_equals_zero_l1111_111121


namespace NUMINAMATH_CALUDE_ru_length_is_8_25_l1111_111117

/-- Triangle PQR with given side lengths and specific geometric constructions -/
structure SpecialTriangle where
  /-- Side length PQ -/
  pq : ℝ
  /-- Side length QR -/
  qr : ℝ
  /-- Side length RP -/
  rp : ℝ
  /-- Point S on QR where the angle bisector of ∠PQR intersects QR -/
  s : ℝ × ℝ
  /-- Point T on the circumcircle of PQR where the angle bisector of ∠PQR intersects (T ≠ P) -/
  t : ℝ × ℝ
  /-- Point U on PQ where the circumcircle of PST intersects (U ≠ P) -/
  u : ℝ × ℝ
  /-- PQ = 13 -/
  h_pq : pq = 13
  /-- QR = 30 -/
  h_qr : qr = 30
  /-- RP = 26 -/
  h_rp : rp = 26
  /-- S is on QR -/
  h_s_on_qr : s.1 + s.2 = qr
  /-- T is on the circumcircle of PQR -/
  h_t_on_circumcircle : True  -- placeholder
  /-- U is on PQ -/
  h_u_on_pq : u.1 + u.2 = pq
  /-- T ≠ P -/
  h_t_ne_p : t ≠ (0, 0)
  /-- U ≠ P -/
  h_u_ne_p : u ≠ (0, 0)

/-- The length of RU in the special triangle construction -/
def ruLength (tri : SpecialTriangle) : ℝ := sorry

/-- Theorem stating that RU = 8.25 in the special triangle construction -/
theorem ru_length_is_8_25 (tri : SpecialTriangle) : ruLength tri = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_ru_length_is_8_25_l1111_111117


namespace NUMINAMATH_CALUDE_fifth_month_sale_is_13562_l1111_111184

/-- The sale amount in the fifth month given the conditions of the problem -/
def fifth_month_sale (first_month : ℕ) (second_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (sixth_month : ℕ) (average : ℕ) : ℕ :=
  average * 6 - (first_month + second_month + third_month + fourth_month + sixth_month)

/-- Theorem stating that the fifth month sale is 13562 given the problem conditions -/
theorem fifth_month_sale_is_13562 :
  fifth_month_sale 6435 6927 6855 7230 5591 6600 = 13562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_is_13562_l1111_111184


namespace NUMINAMATH_CALUDE_solution_set_iff_a_half_l1111_111106

theorem solution_set_iff_a_half (a : ℝ) :
  (∀ x : ℝ, (a * x) / (x - 1) < 1 ↔ x < 1 ∨ x > 2) ↔ a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_iff_a_half_l1111_111106


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l1111_111110

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * (x - 3)^2 + 16 * (y + 2)^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (3, -2)

-- Define the lengths of semi-major and semi-minor axes
def a : ℝ := 4
def b : ℝ := 2

-- Define an endpoint of the major axis
def C : ℝ × ℝ := (center.1 + a, center.2)

-- Define an endpoint of the minor axis
def D : ℝ × ℝ := (center.1, center.2 + b)

-- Theorem statement
theorem ellipse_axis_endpoints_distance : 
  let distance := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  distance = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l1111_111110


namespace NUMINAMATH_CALUDE_julio_earnings_l1111_111157

/-- Calculates the total earnings for Julio over 3 weeks --/
def total_earnings (commission_rate : ℕ) (first_week_customers : ℕ) (salary : ℕ) (bonus : ℕ) : ℕ :=
  let second_week_customers := 2 * first_week_customers
  let third_week_customers := 3 * first_week_customers
  let total_customers := first_week_customers + second_week_customers + third_week_customers
  let commission := commission_rate * total_customers
  salary + commission + bonus

/-- Theorem stating that Julio's total earnings for 3 weeks is $760 --/
theorem julio_earnings : 
  total_earnings 1 35 500 50 = 760 := by
  sorry

end NUMINAMATH_CALUDE_julio_earnings_l1111_111157


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_minus_2017_l1111_111105

theorem divisibility_of_sum_of_squares_minus_2017 :
  ∀ n : ℕ, ∃ x y : ℤ, (n : ℤ) ∣ (x^2 + y^2 - 2017) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_squares_minus_2017_l1111_111105


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1111_111127

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (n : ℝ) / 2 * (a 1 + a n)) →  -- sum formula
  3 * a 5 - a 1 = 10 →  -- given condition
  S 13 = 117 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1111_111127


namespace NUMINAMATH_CALUDE_min_questions_required_l1111_111178

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents a box containing a ball -/
structure Box where
  ball : Color

/-- Represents the state of the boxes -/
structure BoxState where
  boxes : Vector Box 2004
  white_count : Nat
  white_count_even : Even white_count

/-- Represents a question about two boxes -/
structure Question where
  box1 : Fin 2004
  box2 : Fin 2004
  box1_ne_box2 : box1 ≠ box2

/-- The result of asking a question -/
def ask_question (state : BoxState) (q : Question) : Bool :=
  match state.boxes[q.box1].ball, state.boxes[q.box2].ball with
  | Color.White, _ => true
  | _, Color.White => true
  | _, _ => false

/-- A strategy for asking questions -/
def Strategy := Nat → Question

/-- Checks if a strategy is successful for a given state -/
def strategy_successful (state : BoxState) (strategy : Strategy) : Prop :=
  ∃ n : Nat, ∃ i j : Fin 2004,
    i ≠ j ∧
    state.boxes[i].ball = Color.White ∧
    state.boxes[j].ball = Color.White ∧
    (∀ k < n, ask_question state (strategy k) = true)

/-- The main theorem stating the minimum number of questions required -/
theorem min_questions_required :
  ∀ (strategy : Strategy),
  (∀ state : BoxState, strategy_successful state strategy) →
  (∃ n : Nat, ∀ k, strategy k = strategy n → k ≥ 4005) :=
sorry

end NUMINAMATH_CALUDE_min_questions_required_l1111_111178


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l1111_111108

theorem quadratic_equation_problem (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + 2*m*x₁ + m^2 - m + 2 = 0 ∧
    x₂^2 + 2*m*x₂ + m^2 - m + 2 = 0 ∧
    x₁ + x₂ + x₁ * x₂ = 2) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l1111_111108


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1111_111154

-- Define the quadratic function f
def f (x : ℝ) := 2 * x^2 - 4 * x + 3

-- State the theorem
theorem quadratic_function_properties :
  (f 1 = 1) ∧
  (∀ x, f (x + 1) - f x = 4 * x - 2) ∧
  (∀ a, (∃ x y, 2 * a ≤ x ∧ x < y ∧ y ≤ a + 1 ∧ f x > f y ∧ ∃ z, x < z ∧ z < y ∧ f z > f x)
    ↔ (0 < a ∧ a < 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1111_111154


namespace NUMINAMATH_CALUDE_star_property_l1111_111137

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.four
  | Element.two, Element.three => Element.three
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.one
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.four

theorem star_property : 
  star (star Element.three Element.two) (star Element.two Element.one) = Element.four := by
  sorry

end NUMINAMATH_CALUDE_star_property_l1111_111137


namespace NUMINAMATH_CALUDE_fraction_problem_l1111_111140

theorem fraction_problem (n : ℝ) (h : n = 180) : ∃ f : ℝ, f * (1/3 * 1/5 * n) + 6 = 1/15 * n ∧ f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1111_111140


namespace NUMINAMATH_CALUDE_tony_rollercoasters_l1111_111152

/-- The number of rollercoasters Tony went on -/
def num_rollercoasters : ℕ := 5

/-- The speeds of the rollercoasters Tony went on -/
def rollercoaster_speeds : List ℝ := [50, 62, 73, 70, 40]

/-- The average speed of all rollercoasters Tony went on -/
def average_speed : ℝ := 59

/-- Theorem stating that the number of rollercoasters Tony went on is correct -/
theorem tony_rollercoasters :
  num_rollercoasters = rollercoaster_speeds.length ∧
  (rollercoaster_speeds.sum / num_rollercoasters : ℝ) = average_speed := by
  sorry

end NUMINAMATH_CALUDE_tony_rollercoasters_l1111_111152


namespace NUMINAMATH_CALUDE_max_negative_integers_l1111_111153

theorem max_negative_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) :
  ∃ (w : ℕ), w ≤ 4 ∧
  ∀ (n : ℕ), (∃ (s : Finset (Fin 6)), s.card = n ∧
    (∀ i ∈ s, match i with
      | 0 => a < 0
      | 1 => b < 0
      | 2 => c < 0
      | 3 => d < 0
      | 4 => e < 0
      | 5 => f < 0
    )) → n ≤ w :=
by sorry

end NUMINAMATH_CALUDE_max_negative_integers_l1111_111153


namespace NUMINAMATH_CALUDE_division_expression_equality_l1111_111169

theorem division_expression_equality : 180 / (8 + 9 * 3 - 4) = 180 / 31 := by
  sorry

end NUMINAMATH_CALUDE_division_expression_equality_l1111_111169


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1111_111163

theorem inscribed_square_area (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0) 
  (h4 : a * b = x^2) (h5 : a = 34) (h6 : b = 66) : x^2 = 2244 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1111_111163


namespace NUMINAMATH_CALUDE_inequality_proof_l1111_111146

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1111_111146


namespace NUMINAMATH_CALUDE_printer_time_relationship_l1111_111103

/-- Represents a printer's capability to print leaflets -/
structure Printer :=
  (time : ℝ)  -- Time taken to print 800 leaflets

/-- Represents a system of two printers -/
structure PrinterSystem :=
  (printer1 : Printer)
  (printer2 : Printer)
  (combined_time : ℝ)  -- Time taken by both printers together to print 800 leaflets

/-- Theorem stating the relationship between individual printer times and combined time -/
theorem printer_time_relationship (system : PrinterSystem) 
    (h1 : system.printer1.time = 12)
    (h2 : system.combined_time = 3) :
    (1 / system.printer1.time) + (1 / system.printer2.time) = (1 / system.combined_time) :=
  sorry

end NUMINAMATH_CALUDE_printer_time_relationship_l1111_111103


namespace NUMINAMATH_CALUDE_fraction_multiple_l1111_111175

theorem fraction_multiple (numerator denominator : ℕ) : 
  denominator = 5 →
  numerator = denominator + 4 →
  (numerator + 6) / denominator = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_multiple_l1111_111175


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1111_111174

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (10 + n) = 8 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1111_111174


namespace NUMINAMATH_CALUDE_expanded_dining_area_total_l1111_111155

/-- The total area of an expanded outdoor dining area consisting of a rectangular section
    with an area of 35 square feet and a semi-circular section with a radius of 4 feet
    is equal to 35 + 8π square feet. -/
theorem expanded_dining_area_total (rectangular_area : ℝ) (semi_circle_radius : ℝ) :
  rectangular_area = 35 ∧ semi_circle_radius = 4 →
  rectangular_area + (1/2 * π * semi_circle_radius^2) = 35 + 8*π := by
  sorry

end NUMINAMATH_CALUDE_expanded_dining_area_total_l1111_111155


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l1111_111115

noncomputable def triangle_side_b (a : ℝ) (A B : ℝ) : ℝ :=
  2 * Real.sqrt 6

theorem triangle_side_b_value (a : ℝ) (A B : ℝ) 
  (h1 : a = 3)
  (h2 : B = 2 * A)
  (h3 : Real.cos A = Real.sqrt 6 / 3) :
  triangle_side_b a A B = 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l1111_111115


namespace NUMINAMATH_CALUDE_cost_of_500_pencils_is_25_dollars_l1111_111131

/-- The cost of 500 pencils in dollars -/
def cost_of_500_pencils : ℚ :=
  let cost_per_pencil : ℚ := 5 / 100  -- 5 cents in dollars
  let number_of_pencils : ℕ := 500
  cost_per_pencil * number_of_pencils

theorem cost_of_500_pencils_is_25_dollars :
  cost_of_500_pencils = 25 := by sorry

end NUMINAMATH_CALUDE_cost_of_500_pencils_is_25_dollars_l1111_111131


namespace NUMINAMATH_CALUDE_square_of_product_l1111_111123

theorem square_of_product (x : ℝ) : (3 * x)^2 = 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_l1111_111123


namespace NUMINAMATH_CALUDE_system_solution_l1111_111150

theorem system_solution (x y : ℝ) :
  (2 * x + 3 * y = 14) → (x + 4 * y = 11) → (x - y = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1111_111150


namespace NUMINAMATH_CALUDE_quadratic_positivity_set_l1111_111116

/-- Given a quadratic function with zeros at -2 and 3, prove its positivity set -/
theorem quadratic_positivity_set 
  (y : ℝ → ℝ) 
  (h1 : ∀ x, y x = x^2 + b*x + c) 
  (h2 : y (-2) = 0) 
  (h3 : y 3 = 0) :
  {x : ℝ | y x > 0} = {x | x < -2 ∨ x > 3} :=
sorry

end NUMINAMATH_CALUDE_quadratic_positivity_set_l1111_111116


namespace NUMINAMATH_CALUDE_racket_sales_revenue_l1111_111113

theorem racket_sales_revenue 
  (average_price : ℝ) 
  (pairs_sold : ℕ) 
  (h1 : average_price = 9.8) 
  (h2 : pairs_sold = 75) :
  average_price * (pairs_sold : ℝ) = 735 := by
  sorry

end NUMINAMATH_CALUDE_racket_sales_revenue_l1111_111113


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l1111_111130

theorem jelly_bean_probability (red green yellow blue : ℕ) 
  (h_red : red = 7)
  (h_green : green = 9)
  (h_yellow : yellow = 4)
  (h_blue : blue = 10) :
  (red : ℚ) / (red + green + yellow + blue) = 7 / 30 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l1111_111130


namespace NUMINAMATH_CALUDE_dice_throw_pigeonhole_l1111_111176

/-- Represents a throw of four fair six-sided dice -/
def DiceThrow := Fin 4 → Fin 6

/-- The sum of a dice throw -/
def throwSum (t : DiceThrow) : ℕ := (t 0).val + 1 + (t 1).val + 1 + (t 2).val + 1 + (t 3).val + 1

/-- A sequence of dice throws -/
def ThrowSequence (n : ℕ) := Fin n → DiceThrow

theorem dice_throw_pigeonhole :
  ∀ (s : ThrowSequence 22), ∃ (i j : Fin 22), i ≠ j ∧ throwSum (s i) = throwSum (s j) :=
sorry

end NUMINAMATH_CALUDE_dice_throw_pigeonhole_l1111_111176


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l1111_111129

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  -- Given condition
  a * Real.sin C - Real.sqrt 3 * c * Real.cos A = 0 →
  -- Additional conditions
  a = 2 →
  1/2 * b * c * Real.sin A = Real.sqrt 3 →
  -- Conclusion
  A = π/3 ∧ b = 2 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l1111_111129


namespace NUMINAMATH_CALUDE_triangle_area_l1111_111168

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  b = 2 →
  c = 2 * Real.sqrt 2 →
  C = π / 4 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1111_111168


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1111_111165

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 1570000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.57
    exponent := 9
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1111_111165


namespace NUMINAMATH_CALUDE_volume_range_l1111_111145

/-- A rectangular prism with given surface area and sum of edge lengths -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  surface_area_eq : 2 * (a * b + b * c + a * c) = 48
  edge_sum_eq : 4 * (a + b + c) = 36

/-- The volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℝ := p.a * p.b * p.c

/-- Theorem stating the range of possible volumes for the given rectangular prism -/
theorem volume_range (p : RectangularPrism) : 
  16 ≤ volume p ∧ volume p ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_volume_range_l1111_111145


namespace NUMINAMATH_CALUDE_chad_sandwiches_l1111_111135

/-- The number of crackers Chad uses per sandwich -/
def crackers_per_sandwich : ℕ := 2

/-- The number of sleeves in a box of crackers -/
def sleeves_per_box : ℕ := 4

/-- The number of crackers in each sleeve -/
def crackers_per_sleeve : ℕ := 28

/-- The number of boxes of crackers -/
def num_boxes : ℕ := 5

/-- The number of nights 5 boxes of crackers last Chad -/
def num_nights : ℕ := 56

/-- The number of sandwiches Chad has each night -/
def sandwiches_per_night : ℕ := 5

theorem chad_sandwiches :
  sandwiches_per_night * crackers_per_sandwich * num_nights =
  num_boxes * sleeves_per_box * crackers_per_sleeve :=
sorry

end NUMINAMATH_CALUDE_chad_sandwiches_l1111_111135


namespace NUMINAMATH_CALUDE_expected_rolls_in_non_leap_year_l1111_111192

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six
  | Seven
  | Eight

/-- The probability of each outcome on a fair 8-sided die -/
def dieProbability : DieOutcome → ℚ
  | DieOutcome.One => 1/8
  | DieOutcome.Two => 1/8
  | DieOutcome.Three => 1/8
  | DieOutcome.Four => 1/8
  | DieOutcome.Five => 1/8
  | DieOutcome.Six => 1/8
  | DieOutcome.Seven => 1/8
  | DieOutcome.Eight => 1/8

/-- The expected number of rolls on a single day -/
noncomputable def expectedRollsPerDay : ℚ := 8/7

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- The theorem to prove -/
theorem expected_rolls_in_non_leap_year :
  (expectedRollsPerDay * daysInNonLeapYear : ℚ) = 417.14 := by
  sorry


end NUMINAMATH_CALUDE_expected_rolls_in_non_leap_year_l1111_111192


namespace NUMINAMATH_CALUDE_more_philosophers_than_mathematicians_l1111_111120

theorem more_philosophers_than_mathematicians
  (m p : ℕ+)
  (h : (m : ℚ) / 7 = (p : ℚ) / 9) :
  p > m :=
sorry

end NUMINAMATH_CALUDE_more_philosophers_than_mathematicians_l1111_111120


namespace NUMINAMATH_CALUDE_problem_solution_l1111_111185

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

theorem problem_solution :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) ∧
  (∀ k : ℝ, {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1} ⊆ A → k > 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1111_111185


namespace NUMINAMATH_CALUDE_trailing_zeros_of_square_l1111_111128

theorem trailing_zeros_of_square : ∃ n : ℕ, (10^11 - 2)^2 = n * 10^10 ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_square_l1111_111128


namespace NUMINAMATH_CALUDE_multiply_divide_sqrt_equation_l1111_111101

theorem multiply_divide_sqrt_equation (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 1.3333333333333333) :
  (x * y) / 3 = x^2 ↔ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_sqrt_equation_l1111_111101


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1111_111104

/-- An isosceles triangle with two sides of length 6 and one side of length 3 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b : ℝ),
      a = 6 ∧
      b = 3 ∧
      (a = a ∧ b ≤ a + a) ∧  -- Triangle inequality
      perimeter = a + a + b ∧
      perimeter = 15

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 15 := by
  sorry

#check isosceles_triangle_perimeter
#check isosceles_triangle_perimeter_proof

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1111_111104


namespace NUMINAMATH_CALUDE_beatrice_tv_shopping_l1111_111171

theorem beatrice_tv_shopping (x : ℕ) 
  (h1 : x > 0)  -- Beatrice looked at some TVs in the first store
  (h2 : 42 = x + 3*x + 10) : -- Total TVs = First store + Online store + Auction site
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_beatrice_tv_shopping_l1111_111171


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_intersection_A_complement_B_l1111_111189

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 3} := by sorry

-- Theorem for (∁ᵤA) ∩ B
theorem intersection_complement_A_B : (Aᶜ : Set ℝ) ∩ B = {x | 3 ≤ x ∧ x < 5} := by sorry

-- Theorem for A ∩ (∁ᵤB)
theorem intersection_A_complement_B : A ∩ (Bᶜ : Set ℝ) = {x | -1 < x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_intersection_A_complement_B_l1111_111189


namespace NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l1111_111196

theorem permutations_of_seven_distinct_objects : Nat.factorial 7 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_seven_distinct_objects_l1111_111196


namespace NUMINAMATH_CALUDE_f_always_above_g_iff_m_less_than_5_l1111_111151

/-- The function f(x) = |x-2| -/
def f (x : ℝ) : ℝ := |x - 2|

/-- The function g(x) = -|x+3| + m -/
def g (x m : ℝ) : ℝ := -|x + 3| + m

/-- Theorem stating that f(x) > g(x) for all x if and only if m < 5 -/
theorem f_always_above_g_iff_m_less_than_5 :
  (∀ x : ℝ, f x > g x m) ↔ m < 5 := by sorry

end NUMINAMATH_CALUDE_f_always_above_g_iff_m_less_than_5_l1111_111151


namespace NUMINAMATH_CALUDE_train_journey_time_l1111_111111

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  (4/5 * usual_speed) * (usual_time + 3/4) = usual_speed * usual_time → 
  usual_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l1111_111111


namespace NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l1111_111170

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z - Complex.I * 7) = 17) :
  ∃ (w : ℂ), Complex.abs (z - 15) + Complex.abs (z - Complex.I * 7) = 17 ∧ Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 105 / 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l1111_111170


namespace NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l1111_111134

/-- The number of diagonals from one vertex in a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals_from_vertex :
  diagonals_from_vertex decagon_sides = 7 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l1111_111134


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1111_111183

theorem trigonometric_simplification (A : ℝ) (h : 0 < A ∧ A < π / 2) :
  (2 + 2 * (Real.cos A / Real.sin A) - 3 * (1 / Real.sin A)) *
  (3 + 2 * (Real.sin A / Real.cos A) + 1 / Real.cos A) = 11 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1111_111183


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1111_111191

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {1, 3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl A ∩ B) = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1111_111191


namespace NUMINAMATH_CALUDE_reflection_line_equation_l1111_111194

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The reflection of a point over a line -/
def reflect (p : Point) (l : Line) : Point :=
  sorry

/-- The line of reflection given three points and their reflections -/
def reflectionLine (p q r p' q' r' : Point) : Line :=
  sorry

/-- Theorem stating that the line of reflection for the given points has the equation y = (3/5)x + 3/5 -/
theorem reflection_line_equation :
  let p := Point.mk (-2) 1
  let q := Point.mk 3 5
  let r := Point.mk 6 3
  let p' := Point.mk (-4) (-1)
  let q' := Point.mk 1 1
  let r' := Point.mk 4 (-1)
  let l := reflectionLine p q r p' q' r'
  l.slope = 3/5 ∧ l.intercept = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l1111_111194


namespace NUMINAMATH_CALUDE_tan_product_equals_two_l1111_111193

theorem tan_product_equals_two (α β : Real) 
  (h1 : Real.sin α = 2 * Real.sin β) 
  (h2 : Real.sin (α + β) * Real.tan (α - β) = 1) : 
  Real.tan α * Real.tan β = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_equals_two_l1111_111193


namespace NUMINAMATH_CALUDE_tangent_line_condition_l1111_111133

/-- The function f(x) = 2x - a ln x has a tangent line y = x + 1 at the point (1, f(1)) if and only if a = 1 -/
theorem tangent_line_condition (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = 2*x - a * Real.log x) ∧ 
   (∃ g : ℝ → ℝ, (∀ x, g x = x + 1) ∧ 
    (∀ h : ℝ → ℝ, HasDerivAt f (g 1 - f 1) 1 → h = g))) ↔ 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l1111_111133


namespace NUMINAMATH_CALUDE_age_calculation_l1111_111149

/-- Given a two-digit birth year satisfying certain conditions, prove the person's age in 1955 --/
theorem age_calculation (x y : ℕ) (h : 10 * x + y + 4 = 43) : 1955 - (1900 + 10 * x + y) = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_calculation_l1111_111149


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1111_111100

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 74 ∧ n % 7 = 3 ∧ ∀ m : ℕ, m < 74 ∧ m % 7 = 3 → m ≤ n → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1111_111100


namespace NUMINAMATH_CALUDE_square_of_negative_sum_l1111_111136

theorem square_of_negative_sum (x y : ℝ) : (-x - y)^2 = x^2 + 2*x*y + y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sum_l1111_111136


namespace NUMINAMATH_CALUDE_cube_root_simplification_l1111_111160

theorem cube_root_simplification :
  ∃ (a b : ℕ+), (a.val : ℝ) * (b.val : ℝ)^(1/3 : ℝ) = (2^11 * 3^8 : ℝ)^(1/3 : ℝ) ∧ 
  a.val = 72 ∧ b.val = 36 := by
sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l1111_111160


namespace NUMINAMATH_CALUDE_bleach_time_is_correct_l1111_111187

/-- Represents the hair dyeing process with given time constraints -/
def HairDyeingProcess (total_time bleach_time : ℝ) : Prop :=
  bleach_time > 0 ∧
  total_time = bleach_time + (4 * bleach_time) + (1/3 * bleach_time)

/-- Theorem stating that given the constraints, the bleaching time is 1.875 hours -/
theorem bleach_time_is_correct (total_time : ℝ) 
  (h : total_time = 10) : 
  ∃ (bleach_time : ℝ), HairDyeingProcess total_time bleach_time ∧ bleach_time = 1.875 := by
  sorry

end NUMINAMATH_CALUDE_bleach_time_is_correct_l1111_111187


namespace NUMINAMATH_CALUDE_sales_tax_calculation_l1111_111138

-- Define the total spent
def total_spent : ℝ := 40

-- Define the tax rate
def tax_rate : ℝ := 0.06

-- Define the cost of tax-free items
def tax_free_cost : ℝ := 34.7

-- Theorem to prove
theorem sales_tax_calculation :
  let taxable_cost := total_spent - tax_free_cost
  let sales_tax := taxable_cost * tax_rate / (1 + tax_rate)
  sales_tax = 0.3 := by sorry

end NUMINAMATH_CALUDE_sales_tax_calculation_l1111_111138


namespace NUMINAMATH_CALUDE_sum_f_negative_l1111_111156

/-- A monotonically decreasing odd function. -/
def MonoDecreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧ (∀ x, f (-x) = -f x)

theorem sum_f_negative
  (f : ℝ → ℝ)
  (h_f : MonoDecreasingOddFunction f)
  (x₁ x₂ x₃ : ℝ)
  (h₁₂ : x₁ + x₂ > 0)
  (h₂₃ : x₂ + x₃ > 0)
  (h₃₁ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l1111_111156


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l1111_111112

theorem relationship_between_x_and_y (x y m : ℝ) 
  (hx : x = 3 - m) (hy : y = 2*m + 1) : 2*x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l1111_111112


namespace NUMINAMATH_CALUDE_cubic_inequality_l1111_111195

theorem cubic_inequality (x : ℝ) : x > 1 → 2 * x^3 > x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1111_111195


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l1111_111143

/-- The first term of an infinite geometric series with common ratio 1/4 and sum 80 is 60. -/
theorem first_term_of_geometric_series : ∀ (a : ℝ),
  (∑' n, a * (1/4)^n = 80) → a = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l1111_111143


namespace NUMINAMATH_CALUDE_highest_affordable_price_is_8_l1111_111164

/-- The highest whole-dollar price per shirt Alec can afford -/
def highest_affordable_price (total_budget : ℕ) (num_shirts : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) : ℕ :=
  sorry

/-- The proposition to be proved -/
theorem highest_affordable_price_is_8 :
  highest_affordable_price 180 20 5 (8/100) = 8 := by
  sorry

end NUMINAMATH_CALUDE_highest_affordable_price_is_8_l1111_111164


namespace NUMINAMATH_CALUDE_eight_couples_handshakes_l1111_111132

/-- The number of handshakes in a gathering of couples where each person
    shakes hands with everyone except their spouse -/
def handshakes (n : ℕ) : ℕ :=
  (2 * n) * (2 * n - 2) / 2

/-- Theorem: In a gathering of 8 couples, if each person shakes hands once
    with everyone except their spouse, the total number of handshakes is 112 -/
theorem eight_couples_handshakes :
  handshakes 8 = 112 := by
  sorry

#eval handshakes 8  -- Should output 112

end NUMINAMATH_CALUDE_eight_couples_handshakes_l1111_111132


namespace NUMINAMATH_CALUDE_khali_snow_volume_l1111_111180

/-- Calculates the total volume of snow to be shoveled given sidewalk dimensions and snow depths -/
def total_snow_volume (length width initial_depth additional_depth : ℚ) : ℚ :=
  length * width * (initial_depth + additional_depth)

/-- Proves that the total snow volume for Khali's sidewalk is 90 cubic feet -/
theorem khali_snow_volume :
  let length : ℚ := 30
  let width : ℚ := 3
  let initial_depth : ℚ := 3/4
  let additional_depth : ℚ := 1/4
  total_snow_volume length width initial_depth additional_depth = 90 := by
  sorry

end NUMINAMATH_CALUDE_khali_snow_volume_l1111_111180


namespace NUMINAMATH_CALUDE_triangle_inequality_incenter_l1111_111188

/-- Given a triangle ABC with sides a, b, c and a point P inside the triangle with distances
    r₁, r₂, r₃ to the sides respectively, prove that (a/r₁ + b/r₂ + c/r₃) ≥ (a + b + c)²/(2S),
    where S is the area of triangle ABC, and equality holds iff P is the incenter. -/
theorem triangle_inequality_incenter (a b c r₁ r₂ r₃ S : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ S > 0)
  (h_area : a * r₁ + b * r₂ + c * r₃ = 2 * S) :
  a / r₁ + b / r₂ + c / r₃ ≥ (a + b + c)^2 / (2 * S) ∧
  (a / r₁ + b / r₂ + c / r₃ = (a + b + c)^2 / (2 * S) ↔ r₁ = r₂ ∧ r₂ = r₃) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_incenter_l1111_111188


namespace NUMINAMATH_CALUDE_sector_central_angle_l1111_111159

/-- Given a sector with circumference 6 and area 2, its central angle in radians is either 1 or 4. -/
theorem sector_central_angle (r l : ℝ) (h1 : 2*r + l = 6) (h2 : (1/2)*l*r = 2) :
  l/r = 1 ∨ l/r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1111_111159


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l1111_111142

def z : ℂ := (-8 + Complex.I) * Complex.I

theorem z_in_third_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l1111_111142


namespace NUMINAMATH_CALUDE_one_switch_determines_light_l1111_111107

/-- Represents the state of a switch -/
inductive SwitchState
| Position1
| Position2
| Position3

/-- Represents a light bulb -/
inductive Light
| Bulb1
| Bulb2
| Bulb3

/-- Configuration of all switches -/
def SwitchConfig (n : ℕ) := Fin n → SwitchState

/-- Function that determines which light is on given a switch configuration -/
def lightOn (n : ℕ) (config : SwitchConfig n) : Light := sorry

theorem one_switch_determines_light (n : ℕ) :
  (∀ (config : SwitchConfig n), ∃! (l : Light), lightOn n config = l) →
  (∀ (config1 config2 : SwitchConfig n), 
    (∀ i, config1 i ≠ config2 i) → lightOn n config1 ≠ lightOn n config2) →
  ∃ (k : Fin n), ∀ (config1 config2 : SwitchConfig n),
    (∀ (i : Fin n), i ≠ k → config1 i = config2 i) →
    (config1 k = config2 k → lightOn n config1 = lightOn n config2) :=
sorry

end NUMINAMATH_CALUDE_one_switch_determines_light_l1111_111107


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1111_111109

open Real

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  (∀ φ, 0 < φ ∧ φ < π / 2 →
    3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ ≤ 3 * cos φ + 2 / sin φ + 2 * sqrt 2 * tan φ) ∧
  3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ = 7 * sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1111_111109


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l1111_111181

/-- Fred's earnings over the weekend -/
def fred_earnings (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Theorem stating Fred's earnings -/
theorem fred_weekend_earnings :
  fred_earnings 19 40 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l1111_111181


namespace NUMINAMATH_CALUDE_rectangle_length_decrease_l1111_111167

theorem rectangle_length_decrease (b : ℝ) (x : ℝ) : 
  2 * b = 33.333333333333336 →
  (2 * b - x) * (b + 4) = 2 * b^2 + 75 →
  x = 2.833333333333336 := by sorry

end NUMINAMATH_CALUDE_rectangle_length_decrease_l1111_111167


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l1111_111118

theorem arctan_tan_difference (θ₁ θ₂ : Real) (h₁ : θ₁ = 75 * π / 180) (h₂ : θ₂ = 35 * π / 180) :
  Real.arctan (Real.tan θ₁ - 2 * Real.tan θ₂) = 15 * π / 180 := by
  sorry

#check arctan_tan_difference

end NUMINAMATH_CALUDE_arctan_tan_difference_l1111_111118


namespace NUMINAMATH_CALUDE_sports_league_games_l1111_111114

theorem sports_league_games (total_teams : Nat) (divisions : Nat) (teams_per_division : Nat)
  (intra_division_games : Nat) (inter_division_games : Nat) :
  total_teams = divisions * teams_per_division →
  divisions = 3 →
  teams_per_division = 4 →
  intra_division_games = 3 →
  inter_division_games = 1 →
  (total_teams * (((teams_per_division - 1) * intra_division_games) +
    ((total_teams - teams_per_division) * inter_division_games))) / 2 = 102 := by
  sorry

end NUMINAMATH_CALUDE_sports_league_games_l1111_111114


namespace NUMINAMATH_CALUDE_unique_pairs_count_l1111_111190

theorem unique_pairs_count (num_teenagers num_adults : ℕ) : 
  num_teenagers = 12 → num_adults = 8 → 
  (num_teenagers.choose 2) + (num_adults.choose 2) + (num_teenagers * num_adults) = 190 := by
  sorry

end NUMINAMATH_CALUDE_unique_pairs_count_l1111_111190


namespace NUMINAMATH_CALUDE_max_value_rational_function_l1111_111122

theorem max_value_rational_function (x : ℝ) :
  x^4 / (x^8 + 2*x^6 + 4*x^4 + 8*x^2 + 16) ≤ 1/20 ∧
  ∃ y : ℝ, y^4 / (y^8 + 2*y^6 + 4*y^4 + 8*y^2 + 16) = 1/20 :=
by sorry

end NUMINAMATH_CALUDE_max_value_rational_function_l1111_111122


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1111_111102

theorem quadratic_equation_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 = 0 → x < 0) → 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*a*x + 1 = 0 ∧ y^2 + 2*a*y + 1 = 0) → 
  a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1111_111102


namespace NUMINAMATH_CALUDE_base_k_representation_of_5_29_l1111_111147

theorem base_k_representation_of_5_29 (k : ℕ) : k > 0 → (
  (5 : ℚ) / 29 = (k + 3 : ℚ) / (k^2 - 1) ↔ k = 8
) := by sorry

end NUMINAMATH_CALUDE_base_k_representation_of_5_29_l1111_111147


namespace NUMINAMATH_CALUDE_tv_selection_theorem_l1111_111179

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of Type A televisions -/
def typeA : ℕ := 4

/-- The number of Type B televisions -/
def typeB : ℕ := 5

/-- The total number of televisions to be chosen -/
def totalChosen : ℕ := 3

/-- The number of ways to choose the televisions -/
def waysToChoose : ℕ := choose typeA 2 * choose typeB 1 + choose typeA 1 * choose typeB 2

theorem tv_selection_theorem : waysToChoose = 70 := by sorry

end NUMINAMATH_CALUDE_tv_selection_theorem_l1111_111179


namespace NUMINAMATH_CALUDE_base_7_representation_of_864_base_7_correctness_l1111_111124

/-- Converts a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_7_representation_of_864 :
  toBase7 864 = [2, 3, 4, 3] :=
sorry

theorem base_7_correctness :
  fromBase7 [2, 3, 4, 3] = 864 :=
sorry

end NUMINAMATH_CALUDE_base_7_representation_of_864_base_7_correctness_l1111_111124


namespace NUMINAMATH_CALUDE_quadratic_one_solution_positive_n_for_one_solution_l1111_111177

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) ↔ n = 20 ∨ n = -20 := by
  sorry

theorem positive_n_for_one_solution (n : ℝ) :
  n > 0 ∧ (∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_positive_n_for_one_solution_l1111_111177


namespace NUMINAMATH_CALUDE_function_properties_l1111_111126

def f (x : ℝ) : ℝ := |2*x - 1| + |x - 2|

theorem function_properties :
  (∀ k : ℝ, (∀ x₀ : ℝ, f x₀ ≥ |k + 3| - |k - 2|) ↔ k ≤ 1/4) ∧
  (∀ m n : ℝ, (∀ x : ℝ, f x ≥ 1/m + 1/n) → m + n ≥ 8/3) ∧
  (∃ m n : ℝ, (∀ x : ℝ, f x ≥ 1/m + 1/n) ∧ m + n = 8/3) := by sorry

end NUMINAMATH_CALUDE_function_properties_l1111_111126


namespace NUMINAMATH_CALUDE_planted_fraction_for_specific_plot_l1111_111144

/-- Represents a right triangle plot with an unplanted square at the right angle --/
structure PlotWithUnplantedSquare where
  leg1 : ℝ
  leg2 : ℝ
  unplanted_square_side : ℝ
  shortest_distance_to_hypotenuse : ℝ

/-- Calculates the fraction of the plot that is planted --/
def planted_fraction (plot : PlotWithUnplantedSquare) : ℝ := by sorry

/-- Theorem stating the planted fraction for the given plot dimensions --/
theorem planted_fraction_for_specific_plot :
  let plot : PlotWithUnplantedSquare := {
    leg1 := 5,
    leg2 := 12,
    unplanted_square_side := 3 * 7 / 5,
    shortest_distance_to_hypotenuse := 3
  }
  planted_fraction plot = 412 / 1000 := by sorry

end NUMINAMATH_CALUDE_planted_fraction_for_specific_plot_l1111_111144
