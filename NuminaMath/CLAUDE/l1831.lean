import Mathlib

namespace cs_physics_overlap_l1831_183111

/-- Represents the fraction of students in one club who also attend another club -/
def club_overlap (club1 club2 : Type) : ℚ := sorry

theorem cs_physics_overlap :
  let m := club_overlap Mathematics Physics
  let c := club_overlap Mathematics ComputerScience
  let p := club_overlap Physics Mathematics
  let q := club_overlap Physics ComputerScience
  let r := club_overlap ComputerScience Mathematics
  m = 1/6 ∧ c = 1/8 ∧ p = 1/3 ∧ q = 1/5 ∧ r = 1/7 →
  club_overlap ComputerScience Physics = 4/35 :=
sorry

end cs_physics_overlap_l1831_183111


namespace fraction_equality_l1831_183100

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := a * b - 3 * b^2

-- Define the # operation
def hash_op (a b : ℚ) : ℚ := a + 2 * b - 2 * a * b^2

-- Theorem statement
theorem fraction_equality :
  (at_op 8 3) / (hash_op 8 3) = 3 / 130 := by
  sorry

end fraction_equality_l1831_183100


namespace triangle_angle_c_l1831_183119

theorem triangle_angle_c (A B C : ℝ) : 
  A + B + C = π →  -- Sum of angles in a triangle
  |2 * Real.sin A - 1| + |Real.sqrt 2 / 2 - Real.cos B| = 0 →
  C = 7 * π / 12  -- 105° in radians
:= by sorry

end triangle_angle_c_l1831_183119


namespace max_correct_answers_l1831_183161

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_points = 5 →
  incorrect_points = -2 →
  total_score = 150 →
  ∃ (correct blank incorrect : ℕ),
    correct + blank + incorrect = total_questions ∧
    correct_points * correct + incorrect_points * incorrect = total_score ∧
    correct ≤ 38 ∧
    ∀ (c : ℕ), c > 38 →
      ¬(∃ (b i : ℕ), c + b + i = total_questions ∧
        correct_points * c + incorrect_points * i = total_score) :=
by sorry

end max_correct_answers_l1831_183161


namespace negation_of_existential_proposition_l1831_183136

open Set

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 < 0) ↔ (∀ x : ℝ, x^2 ≥ 0) :=
sorry

end negation_of_existential_proposition_l1831_183136


namespace g_of_three_equals_five_l1831_183110

theorem g_of_three_equals_five (g : ℝ → ℝ) (h : ∀ x, g (x + 2) = 2 * x + 3) :
  g 3 = 5 := by sorry

end g_of_three_equals_five_l1831_183110


namespace weight_of_b_l1831_183153

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) :
  b = 31 := by sorry

end weight_of_b_l1831_183153


namespace mean_height_is_70_l1831_183192

def heights : List ℕ := [58, 59, 60, 62, 63, 65, 65, 68, 70, 71, 71, 72, 76, 76, 78, 79, 79]

theorem mean_height_is_70 : 
  (List.sum heights) / (heights.length : ℚ) = 70 := by
  sorry

end mean_height_is_70_l1831_183192


namespace equation_solution_l1831_183144

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) (h3 : x ≠ 2) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (4 - x) / (x - 2) ↔ 
  x = -3 ∨ x = (1 + Real.sqrt 17) / 2 ∨ x = (1 - Real.sqrt 17) / 2 :=
by sorry

end equation_solution_l1831_183144


namespace tan_alpha_plus_pi_12_l1831_183179

theorem tan_alpha_plus_pi_12 (α : Real) 
  (h : Real.sin α = 3 * Real.sin (α + π/6)) : 
  Real.tan (α + π/12) = 2 * Real.sqrt 3 - 4 := by
  sorry

end tan_alpha_plus_pi_12_l1831_183179


namespace debbys_flour_amount_l1831_183133

/-- Calculates the final amount of flour Debby has -/
def final_flour_amount (initial : ℕ) (used : ℕ) (given : ℕ) (bought : ℕ) : ℕ :=
  initial - used - given + bought

/-- Proves that Debby's final amount of flour is 11 pounds -/
theorem debbys_flour_amount :
  final_flour_amount 12 3 2 4 = 11 := by
  sorry

end debbys_flour_amount_l1831_183133


namespace tshirt_sale_revenue_l1831_183122

/-- Calculates the total money made from selling t-shirts with a discount -/
theorem tshirt_sale_revenue (original_price discount : ℕ) (num_sold : ℕ) :
  original_price = 51 →
  discount = 8 →
  num_sold = 130 →
  (original_price - discount) * num_sold = 5590 :=
by sorry

end tshirt_sale_revenue_l1831_183122


namespace longest_chord_of_circle_with_radius_five_l1831_183105

/-- A circle with a given radius. -/
structure Circle where
  radius : ℝ

/-- The longest chord of a circle is its diameter, which is twice the radius. -/
def longestChordLength (c : Circle) : ℝ := 2 * c.radius

theorem longest_chord_of_circle_with_radius_five :
  ∃ (c : Circle), c.radius = 5 ∧ longestChordLength c = 10 := by
  sorry

end longest_chord_of_circle_with_radius_five_l1831_183105


namespace cubic_equation_solution_l1831_183145

theorem cubic_equation_solution :
  {x : ℝ | x^3 + 6*x^2 + 11*x + 6 = 12} = {-3, -2, -1} := by sorry

end cubic_equation_solution_l1831_183145


namespace max_value_on_circle_l1831_183182

theorem max_value_on_circle : 
  ∀ x y : ℝ, x^2 + y^2 - 6*x + 8 = 0 → x^2 + y^2 ≤ 16 ∧ ∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 6*x₀ + 8 = 0 ∧ x₀^2 + y₀^2 = 16 := by
  sorry

end max_value_on_circle_l1831_183182


namespace bass_strings_l1831_183164

theorem bass_strings (num_basses : ℕ) (num_guitars : ℕ) (num_8string_guitars : ℕ) 
  (guitar_strings : ℕ) (total_strings : ℕ) :
  num_basses = 3 →
  num_guitars = 2 * num_basses →
  guitar_strings = 6 →
  num_8string_guitars = num_guitars - 3 →
  total_strings = 72 →
  ∃ bass_strings : ℕ, 
    bass_strings * num_basses + guitar_strings * num_guitars + 8 * num_8string_guitars = total_strings ∧
    bass_strings = 4 :=
by sorry

end bass_strings_l1831_183164


namespace orange_harvest_orange_harvest_solution_l1831_183196

theorem orange_harvest (discarded : ℕ) (days : ℕ) (remaining : ℕ) : ℕ :=
  let harvested := (remaining + days * discarded) / days
  harvested

theorem orange_harvest_solution :
  orange_harvest 71 51 153 = 74 := by
  sorry

end orange_harvest_orange_harvest_solution_l1831_183196


namespace problem_solution_l1831_183170

open Real

def p : Prop := ∀ x > 0, log x + 4 * x ≥ 3

def q : Prop := ∃ x₀ > 0, 8 * x₀ + 1 / (2 * x₀) ≤ 4

theorem problem_solution : (¬p ∧ q) := by sorry

end problem_solution_l1831_183170


namespace triangle_properties_l1831_183162

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := sorry

theorem triangle_properties (t : Triangle) 
  (h : area t = t.a^2 / 2) : 
  (Real.tan t.A = 2 * t.a^2 / (t.b^2 + t.c^2 - t.a^2)) ∧ 
  (∃ (x : ℝ), x = Real.sqrt 5 ∧ ∀ (y : ℝ), t.c / t.b + t.b / t.c ≤ x) ∧
  (∃ (m : ℝ), ∀ (x : ℝ), m ≤ t.b * t.c / t.a^2) :=
by sorry

end triangle_properties_l1831_183162


namespace agnes_flight_cost_l1831_183128

/-- Represents the cost structure for different transportation modes -/
structure TransportCost where
  busCostPerKm : ℝ
  airplaneCostPerKm : ℝ
  airplaneBookingFee : ℝ

/-- Represents the distances between cities -/
structure CityDistances where
  xToY : ℝ
  xToZ : ℝ

/-- Calculates the cost of an airplane trip -/
def airplaneTripCost (cost : TransportCost) (distance : ℝ) : ℝ :=
  cost.airplaneBookingFee + cost.airplaneCostPerKm * distance

theorem agnes_flight_cost (cost : TransportCost) (distances : CityDistances) :
  cost.busCostPerKm = 0.20 →
  cost.airplaneCostPerKm = 0.12 →
  cost.airplaneBookingFee = 120 →
  distances.xToY = 4500 →
  distances.xToZ = 4000 →
  airplaneTripCost cost distances.xToY = 660 := by
  sorry


end agnes_flight_cost_l1831_183128


namespace smallest_solution_of_equation_l1831_183127

theorem smallest_solution_of_equation :
  ∀ x : ℚ, 3 * (8 * x^2 + 10 * x + 12) = x * (8 * x - 36) →
  x ≥ (-3 : ℚ) :=
by sorry

end smallest_solution_of_equation_l1831_183127


namespace alpha_beta_range_l1831_183160

-- Define the curve E
def curve_E (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the line l1
def line_l1 (k x y : ℝ) : Prop := y = k * (x + 2)

-- Define the intersection points A and B
def intersection_points (x1 y1 x2 y2 k : ℝ) : Prop :=
  curve_E x1 y1 ∧ curve_E x2 y2 ∧ 
  line_l1 k x1 y1 ∧ line_l1 k x2 y2 ∧
  x1 ≠ x2

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the relationship between α, β, and the points
def alpha_beta_relation (α β x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : Prop :=
  α = (1 - x1) / (x3 - 1) ∧
  β = (1 - x2) / (x4 - 1) ∧
  curve_E x3 y3 ∧ curve_E x4 y4

-- Main theorem
theorem alpha_beta_range :
  ∀ (k x1 y1 x2 y2 x3 y3 x4 y4 α β : ℝ),
    0 < k^2 ∧ k^2 < 1/2 →
    intersection_points x1 y1 x2 y2 k →
    alpha_beta_relation α β x1 y1 x2 y2 x3 y3 x4 y4 →
    6 < α + β ∧ α + β < 10 := by
  sorry

end alpha_beta_range_l1831_183160


namespace peony_count_l1831_183120

theorem peony_count (n : ℕ) 
  (h1 : ∃ (s d t : ℕ), n = s + d + t ∧ t = s + 30)
  (h2 : ∃ (x : ℕ), s = 4 * x ∧ d = 2 * x ∧ t = 6 * x) : 
  n = 180 := by
sorry

end peony_count_l1831_183120


namespace dice_sum_theorem_l1831_183103

def Die := Fin 6

def roll_sum (d1 d2 : Die) : ℕ := d1.val + d2.val + 2

def possible_sums : Set ℕ := {n | ∃ (d1 d2 : Die), roll_sum d1 d2 = n}

theorem dice_sum_theorem : possible_sums = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} := by
  sorry

end dice_sum_theorem_l1831_183103


namespace product_of_roots_cubic_l1831_183138

theorem product_of_roots_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 15*x^2 + 75*x - 50
  ∃ a b c : ℝ, (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ a * b * c = 50 :=
by sorry

end product_of_roots_cubic_l1831_183138


namespace range_of_a_l1831_183142

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1 / (x - 1) < 1
def q (x a : ℝ) : Prop := x^2 + (a - 1) * x - a > 0

-- Define the property that p is sufficient but not necessary for q
def p_sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ¬(∀ x, q x a → p x)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, p_sufficient_not_necessary a ↔ -2 < a ∧ a ≤ -1 :=
sorry

end range_of_a_l1831_183142


namespace school_seminar_cost_l1831_183168

/-- Calculates the total amount spent by a school for a teacher seminar with discounts and food allowance. -/
def total_seminar_cost (regular_fee : ℝ) (discount_percent : ℝ) (num_teachers : ℕ) (food_allowance : ℝ) : ℝ :=
  let discounted_fee := regular_fee * (1 - discount_percent)
  let total_seminar_fees := discounted_fee * num_teachers
  let total_food_allowance := food_allowance * num_teachers
  total_seminar_fees + total_food_allowance

/-- Theorem stating the total cost for the school's teacher seminar -/
theorem school_seminar_cost :
  total_seminar_cost 150 0.05 10 10 = 1525 :=
by sorry

end school_seminar_cost_l1831_183168


namespace mrs_hilt_pecan_pies_l1831_183139

/-- The number of pecan pies Mrs. Hilt baked -/
def pecan_pies : ℕ := sorry

/-- The number of apple pies Mrs. Hilt baked -/
def apple_pies : ℕ := 14

/-- The number of rows in the pie arrangement -/
def rows : ℕ := 30

/-- The number of pies in each row -/
def pies_per_row : ℕ := 5

/-- The total number of pies -/
def total_pies : ℕ := rows * pies_per_row

theorem mrs_hilt_pecan_pies :
  pecan_pies = 136 :=
by
  sorry

end mrs_hilt_pecan_pies_l1831_183139


namespace children_events_count_l1831_183134

theorem children_events_count (cupcakes_per_event : ℝ) (total_cupcakes : ℕ) 
  (h1 : cupcakes_per_event = 96.0)
  (h2 : total_cupcakes = 768) :
  (total_cupcakes : ℝ) / cupcakes_per_event = 8 := by
  sorry

end children_events_count_l1831_183134


namespace smallest_nonzero_y_value_l1831_183108

theorem smallest_nonzero_y_value (y : ℝ) : 
  y > 0 ∧ Real.sqrt (6 * y + 3) = 3 * y + 1 → y ≥ Real.sqrt 2 / 3 :=
by sorry

end smallest_nonzero_y_value_l1831_183108


namespace smallest_n_for_integer_roots_l1831_183193

theorem smallest_n_for_integer_roots : ∃ (x y : ℤ),
  x^2 - 91*x + 2014 = 0 ∧ y^2 - 91*y + 2014 = 0 ∧
  (∀ (n : ℕ) (a b : ℤ), n < 91 → (a^2 - n*a + 2014 = 0 ∧ b^2 - n*b + 2014 = 0) → False) :=
by sorry

end smallest_n_for_integer_roots_l1831_183193


namespace bacteria_growth_time_l1831_183101

/-- The time required for bacteria growth under specific conditions -/
theorem bacteria_growth_time (initial_count : ℕ) (final_count : ℕ) (growth_factor : ℕ) (growth_time : ℕ) (total_time : ℕ) : 
  initial_count = 200 →
  final_count = 145800 →
  growth_factor = 3 →
  growth_time = 3 →
  (initial_count * growth_factor ^ (total_time / growth_time) = final_count) →
  total_time = 18 := by
  sorry

end bacteria_growth_time_l1831_183101


namespace bruce_shopping_theorem_l1831_183125

/-- Calculates the remaining money after Bruce's shopping trip. -/
def remaining_money (initial_amount shirt_price num_shirts pants_price : ℕ) : ℕ :=
  initial_amount - (shirt_price * num_shirts + pants_price)

/-- Theorem stating that Bruce has $20 left after his shopping trip. -/
theorem bruce_shopping_theorem :
  remaining_money 71 5 5 26 = 20 := by
  sorry

end bruce_shopping_theorem_l1831_183125


namespace adult_admission_price_l1831_183141

/-- Proves that the admission price for adults is 8 dollars given the specified conditions -/
theorem adult_admission_price
  (total_amount : ℕ)
  (total_tickets : ℕ)
  (children_tickets : ℕ)
  (children_price : ℕ)
  (h1 : total_amount = 201)
  (h2 : total_tickets = 33)
  (h3 : children_tickets = 21)
  (h4 : children_price = 5) :
  (total_amount - children_tickets * children_price) / (total_tickets - children_tickets) = 8 := by
  sorry


end adult_admission_price_l1831_183141


namespace archibald_apple_consumption_l1831_183126

def apples_per_day_first_two_weeks (x : ℝ) : Prop :=
  let first_two_weeks := 14 * x
  let next_three_weeks := 14 * x
  let last_two_weeks := 14 * 3
  let total_apples := 7 * 10
  first_two_weeks + next_three_weeks + last_two_weeks = total_apples

theorem archibald_apple_consumption : 
  ∃ x : ℝ, apples_per_day_first_two_weeks x ∧ x = 1 :=
sorry

end archibald_apple_consumption_l1831_183126


namespace nina_widget_purchase_l1831_183143

theorem nina_widget_purchase (initial_money : ℕ) (initial_widgets : ℕ) (price_reduction : ℕ) 
  (h1 : initial_money = 24)
  (h2 : initial_widgets = 6)
  (h3 : price_reduction = 1)
  : (initial_money / (initial_money / initial_widgets - price_reduction) : ℕ) = 8 := by
  sorry

end nina_widget_purchase_l1831_183143


namespace second_error_greater_l1831_183107

/-- Given two measured lines with their lengths and errors, prove that the absolute error of the second measurement is greater than the first. -/
theorem second_error_greater (length1 length2 error1 error2 : ℝ) 
  (h1 : length1 = 50)
  (h2 : length2 = 200)
  (h3 : error1 = 0.05)
  (h4 : error2 = 0.4) : 
  error2 > error1 := by
  sorry

end second_error_greater_l1831_183107


namespace solution_set_of_inequality_l1831_183158

theorem solution_set_of_inequality (x : ℝ) : 
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 :=
by sorry

end solution_set_of_inequality_l1831_183158


namespace isabella_hair_length_l1831_183159

/-- Calculates the length of hair after a given time period. -/
def hair_length (initial_length : ℝ) (growth_rate : ℝ) (months : ℝ) : ℝ :=
  initial_length + growth_rate * months

/-- Theorem stating that Isabella's hair length after y months is 18 + xy -/
theorem isabella_hair_length (x y : ℝ) :
  hair_length 18 x y = 18 + x * y := by
  sorry

end isabella_hair_length_l1831_183159


namespace hotel_visit_permutations_l1831_183189

def number_of_permutations (n : ℕ) : ℕ := Nat.factorial n

def constrained_permutations (n : ℕ) : ℕ :=
  number_of_permutations n / 4

theorem hotel_visit_permutations :
  constrained_permutations 5 = 30 := by sorry

end hotel_visit_permutations_l1831_183189


namespace movie_theatre_attendance_l1831_183102

theorem movie_theatre_attendance (total_seats : ℕ) (adult_price child_price : ℚ) 
  (total_revenue : ℚ) (h_seats : total_seats = 250) (h_adult_price : adult_price = 6)
  (h_child_price : child_price = 4) (h_revenue : total_revenue = 1124) :
  ∃ (children : ℕ), children = 188 ∧ 
    (∃ (adults : ℕ), adults + children = total_seats ∧
      adult_price * adults + child_price * children = total_revenue) :=
by sorry

end movie_theatre_attendance_l1831_183102


namespace existence_of_special_point_set_l1831_183163

/-- A closed region bounded by a regular polygon -/
structure RegularPolygonRegion where
  vertices : Set (ℝ × ℝ)
  is_regular : Bool
  is_closed : Bool

/-- A set of points in the plane -/
def PointSet := Set (ℝ × ℝ)

/-- Predicate to check if a set of points can be covered by a region -/
def IsCovered (S : PointSet) (C : RegularPolygonRegion) : Prop := sorry

/-- Predicate to check if any n points from a set can be covered by a region -/
def AnyNPointsCovered (S : PointSet) (C : RegularPolygonRegion) (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem existence_of_special_point_set (C : RegularPolygonRegion) (n : ℕ) :
  ∃ (S : PointSet), AnyNPointsCovered S C n ∧ ¬IsCovered S C := by sorry

end existence_of_special_point_set_l1831_183163


namespace inequality_equivalence_l1831_183116

theorem inequality_equivalence (x : ℝ) : 3 * x + 2 < 10 - 2 * x ↔ x < 8 / 5 := by
  sorry

end inequality_equivalence_l1831_183116


namespace square_sum_from_difference_and_product_l1831_183106

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by sorry

end square_sum_from_difference_and_product_l1831_183106


namespace simplify_negative_fraction_power_l1831_183187

theorem simplify_negative_fraction_power : 
  ((-1 : ℝ) / 343) ^ (-(2 : ℝ) / 3) = 49 := by
  sorry

end simplify_negative_fraction_power_l1831_183187


namespace interior_angles_sum_l1831_183194

theorem interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 1440) →
  (180 * ((n + 3) - 2) = 1980) :=
by sorry

end interior_angles_sum_l1831_183194


namespace second_graders_count_l1831_183148

theorem second_graders_count (kindergartners : ℕ) (first_graders : ℕ) (total_students : ℕ) 
  (h1 : kindergartners = 34)
  (h2 : first_graders = 48)
  (h3 : total_students = 120) :
  total_students - (kindergartners + first_graders) = 38 := by
  sorry

end second_graders_count_l1831_183148


namespace system_solution_l1831_183177

theorem system_solution (u v : ℝ) : 
  u + v = 10 ∧ 3 * u - 2 * v = 5 → u = 5 ∧ v = 5 := by
  sorry

end system_solution_l1831_183177


namespace reflection_point_l1831_183180

/-- A function that passes through a given point when shifted -/
def passes_through (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f (a - 1) = b

/-- Reflection of a function across the x-axis -/
def reflect_x (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => -f x

/-- A function passes through a point -/
def function_at_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y

theorem reflection_point (f : ℝ → ℝ) :
  passes_through f 3 2 →
  function_at_point (reflect_x f) 4 (-2) := by
  sorry

end reflection_point_l1831_183180


namespace cafeteria_choices_l1831_183174

theorem cafeteria_choices (num_dishes : ℕ) (num_students : ℕ) : 
  num_dishes = 5 → num_students = 3 → (num_dishes ^ num_students) = 125 := by
  sorry

end cafeteria_choices_l1831_183174


namespace remainder_1743_base12_div_9_l1831_183130

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The base-12 representation of 1743 --/
def num1743Base12 : List Nat := [3, 4, 7, 1]

theorem remainder_1743_base12_div_9 :
  (base12ToBase10 num1743Base12) % 9 = 6 := by
  sorry

end remainder_1743_base12_div_9_l1831_183130


namespace sequence_existence_and_extension_l1831_183132

theorem sequence_existence_and_extension (m : ℕ) (hm : m ≥ 2) :
  (∃ x : ℕ → ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ m → x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) ∧
  (∀ x : ℕ → ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ m → x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) →
    ∃ y : ℤ → ℕ, (∀ k : ℤ, y k * y (m + k) = y (k + 1) * y (m + k - 1) + 1) ∧
               (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2 * m → y i = x i)) :=
by sorry

end sequence_existence_and_extension_l1831_183132


namespace geometric_sequence_sum_l1831_183154

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 1 + a 2 + a 3 = 7) →
  (a 2 + a 3 + a 4 = 14) →
  (a 4 + a 5 + a 6 = 56) :=
by
  sorry

end geometric_sequence_sum_l1831_183154


namespace oblique_triangular_prism_volume_l1831_183171

/-- The volume of an oblique triangular prism with specific properties -/
theorem oblique_triangular_prism_volume (a : ℝ) (ha : a > 0) :
  let base_area := (a^2 * Real.sqrt 3) / 4
  let height := a * Real.sqrt 3 / 2
  base_area * height = (3 * a^3) / 8 := by sorry

end oblique_triangular_prism_volume_l1831_183171


namespace average_age_first_group_l1831_183175

theorem average_age_first_group (total_students : Nat) (avg_age_all : ℝ) 
  (first_group_size second_group_size : Nat) (avg_age_second_group : ℝ) 
  (age_last_student : ℝ) :
  total_students = 15 →
  avg_age_all = 15 →
  first_group_size = 7 →
  second_group_size = 7 →
  avg_age_second_group = 16 →
  age_last_student = 15 →
  (total_students * avg_age_all - second_group_size * avg_age_second_group - age_last_student) / first_group_size = 14 := by
sorry

end average_age_first_group_l1831_183175


namespace exists_n_where_B_less_than_A_l1831_183181

/-- Alphonse's jump function -/
def A (n : ℕ) : ℕ :=
  if n ≥ 8 then A (n - 8) + 1 else n

/-- Beryl's jump function -/
def B (n : ℕ) : ℕ :=
  if n ≥ 7 then B (n - 7) + 1 else n

/-- Theorem stating the existence of n > 200 where B(n) < A(n) -/
theorem exists_n_where_B_less_than_A :
  ∃ n : ℕ, n > 200 ∧ B n < A n :=
sorry

end exists_n_where_B_less_than_A_l1831_183181


namespace sandy_painting_area_l1831_183195

/-- The area Sandy needs to paint on her bedroom wall -/
def area_to_paint (wall_height wall_length bookshelf_width bookshelf_height : ℝ) : ℝ :=
  wall_height * wall_length - bookshelf_width * bookshelf_height

/-- Theorem stating that Sandy needs to paint 135 square feet -/
theorem sandy_painting_area :
  area_to_paint 10 15 3 5 = 135 := by
  sorry

#eval area_to_paint 10 15 3 5

end sandy_painting_area_l1831_183195


namespace elise_remaining_money_l1831_183117

/-- Calculates the remaining money for Elise --/
def remaining_money (initial saved comic_book puzzle : ℕ) : ℕ :=
  initial + saved - (comic_book + puzzle)

/-- Theorem: Elise's remaining money is $1 --/
theorem elise_remaining_money :
  remaining_money 8 13 2 18 = 1 := by
  sorry

end elise_remaining_money_l1831_183117


namespace opposite_absolute_square_l1831_183109

theorem opposite_absolute_square (x y : ℝ) : 
  (|x - 2| = -(y + 7)^2 ∨ -(x - 2) = (y + 7)^2) → y^x = 49 := by
  sorry

end opposite_absolute_square_l1831_183109


namespace fourth_term_of_geometric_progression_l1831_183165

/-- Given a geometric progression with the first three terms 2^(1/4), 2^(1/8), and 2^(1/16),
    the fourth term is 2^(1/32). -/
theorem fourth_term_of_geometric_progression (a₁ a₂ a₃ a₄ : ℝ) : 
  a₁ = 2^(1/4) → a₂ = 2^(1/8) → a₃ = 2^(1/16) → 
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) →
  a₄ = 2^(1/32) := by
sorry

end fourth_term_of_geometric_progression_l1831_183165


namespace third_dimension_of_large_box_l1831_183150

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of small boxes that can fit into a larger box -/
def maxSmallBoxes (largeBox : BoxDimensions) (smallBox : BoxDimensions) : ℕ :=
  (largeBox.length / smallBox.length) * (largeBox.width / smallBox.width) * (largeBox.height / smallBox.height)

theorem third_dimension_of_large_box 
  (largeBox : BoxDimensions) 
  (smallBox : BoxDimensions) 
  (h : ℕ) :
  largeBox.length = 12 ∧ 
  largeBox.width = 14 ∧ 
  largeBox.height = h ∧
  smallBox.length = 3 ∧ 
  smallBox.width = 7 ∧ 
  smallBox.height = 2 ∧
  maxSmallBoxes largeBox smallBox = 64 →
  h = 16 :=
by sorry

end third_dimension_of_large_box_l1831_183150


namespace fraction_relation_l1831_183178

theorem fraction_relation (x y z w : ℚ) 
  (h1 : x / y = 12)
  (h2 : z / y = 4)
  (h3 : z / w = 3 / 4) :
  w / x = 4 / 9 := by
  sorry

end fraction_relation_l1831_183178


namespace hannah_cutting_speed_l1831_183140

/-- The number of strands Hannah can cut per minute -/
def hannah_strands_per_minute : ℕ := 8

/-- The total number of strands of duct tape -/
def total_strands : ℕ := 22

/-- The number of strands Hannah's son can cut per minute -/
def son_strands_per_minute : ℕ := 3

/-- The time it takes to cut all strands (in minutes) -/
def total_time : ℕ := 2

theorem hannah_cutting_speed :
  hannah_strands_per_minute = 8 ∧
  total_strands = 22 ∧
  son_strands_per_minute = 3 ∧
  total_time = 2 ∧
  total_time * (hannah_strands_per_minute + son_strands_per_minute) = total_strands :=
by sorry

end hannah_cutting_speed_l1831_183140


namespace telescope_visual_range_l1831_183129

theorem telescope_visual_range (original_range : ℝ) : 
  (original_range + 1.5 * original_range = 150) → original_range = 60 := by
  sorry

end telescope_visual_range_l1831_183129


namespace unique_solution_for_m_l1831_183104

theorem unique_solution_for_m :
  ∀ (x y m : ℚ),
  (2 * x + y = 3 * m) →
  (x - 4 * y = -2 * m) →
  (y + 2 * m = 1 + x) →
  m = 3 / 5 := by
sorry

end unique_solution_for_m_l1831_183104


namespace kabadi_players_count_l1831_183190

/-- Represents the number of players in different categories -/
structure PlayerCounts where
  total : ℕ
  khoKhoOnly : ℕ
  bothGames : ℕ

/-- Calculates the number of players who play kabadi -/
def kabadiPlayers (counts : PlayerCounts) : ℕ :=
  counts.total - counts.khoKhoOnly + counts.bothGames

/-- Theorem stating the number of kabadi players given the conditions -/
theorem kabadi_players_count (counts : PlayerCounts) 
  (h1 : counts.total = 30)
  (h2 : counts.khoKhoOnly = 20)
  (h3 : counts.bothGames = 5) :
  kabadiPlayers counts = 15 := by
  sorry


end kabadi_players_count_l1831_183190


namespace complement_union_theorem_l1831_183184

universe u

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

theorem complement_union_theorem :
  (Aᶜ ∩ U) ∪ B = {0, 2, 4} := by sorry

end complement_union_theorem_l1831_183184


namespace cross_ratio_preserving_is_projective_l1831_183137

/-- A mapping between two lines -/
structure LineMapping (α : Type*) where
  to_fun : α → α

/-- The cross ratio of four points -/
def cross_ratio {α : Type*} [Field α] (x y z w : α) : α :=
  ((x - z) * (y - w)) / ((x - w) * (y - z))

/-- A mapping preserves cross ratio -/
def preserves_cross_ratio {α : Type*} [Field α] (f : LineMapping α) : Prop :=
  ∀ (x y z w : α), cross_ratio (f.to_fun x) (f.to_fun y) (f.to_fun z) (f.to_fun w) = cross_ratio x y z w

/-- Definition of a projective transformation -/
def is_projective {α : Type*} [Field α] (f : LineMapping α) : Prop :=
  ∃ (a b c d : α), (a * d - b * c ≠ 0) ∧
    (∀ x, f.to_fun x = (a * x + b) / (c * x + d))

/-- Main theorem: A cross-ratio preserving mapping is projective -/
theorem cross_ratio_preserving_is_projective {α : Type*} [Field α] (f : LineMapping α) :
  preserves_cross_ratio f → is_projective f :=
sorry

end cross_ratio_preserving_is_projective_l1831_183137


namespace simplify_polynomial_l1831_183112

theorem simplify_polynomial (r : ℝ) :
  (2 * r^3 + 5 * r^2 + 6 * r - 4) - (r^3 + 9 * r^2 + 4 * r - 7) =
  r^3 - 4 * r^2 + 2 * r + 3 := by
  sorry

end simplify_polynomial_l1831_183112


namespace subset_condition_l1831_183185

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

-- State the theorem
theorem subset_condition (a : ℝ) : B a ⊆ A ↔ a = 0 ∨ a = 1 ∨ a = 2 := by
  sorry

end subset_condition_l1831_183185


namespace complex_division_equality_l1831_183166

theorem complex_division_equality : (3 - I) / (2 + I) = 1 - I := by sorry

end complex_division_equality_l1831_183166


namespace problem_3_l1831_183191

theorem problem_3 (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a^2 + b^2 = 6*a*b) :
  (a + b) / (a - b) = Real.sqrt 2 := by
  sorry

end problem_3_l1831_183191


namespace hexagon_area_from_triangle_l1831_183121

-- Define the regular hexagon ABCDEF
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

-- Define points G, H, I
def G (h : RegularHexagon) : ℝ × ℝ := sorry
def H (h : RegularHexagon) : ℝ × ℝ := sorry
def I (h : RegularHexagon) : ℝ × ℝ := sorry

-- Define the area of a triangle
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the area of a hexagon
def hexagon_area (h : RegularHexagon) : ℝ := sorry

-- Theorem statement
theorem hexagon_area_from_triangle (h : RegularHexagon) :
  triangle_area (G h) (H h) (I h) = 100 → hexagon_area h = 600 := by sorry

end hexagon_area_from_triangle_l1831_183121


namespace sqrt_product_equals_27_l1831_183146

theorem sqrt_product_equals_27 (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (12 * x) * Real.sqrt (18 * x) * Real.sqrt (6 * x) * Real.sqrt (9 * x) = 27) : 
  x = 1 / 2 := by
sorry

end sqrt_product_equals_27_l1831_183146


namespace simplify_expression_l1831_183183

theorem simplify_expression (x : ℝ) : 3*x + 4*x - 2*x + 6*x - 3*x = 8*x := by
  sorry

end simplify_expression_l1831_183183


namespace perpendicular_condition_parallel_condition_l1831_183152

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 2]

-- Define the dot product of two 2D vectors
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity of two 2D vectors
def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

-- Define parallelism of two 2D vectors
def parallel (v w : Fin 2 → ℝ) : Prop := ∃ (c : ℝ), ∀ (i : Fin 2), v i = c * w i

-- Define the vector operations
def add_vectors (v w : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => v i + w i
def scale_vector (k : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => k * v i

-- Theorem statements
theorem perpendicular_condition (k : ℝ) : 
  perpendicular (add_vectors (scale_vector k a) b) (add_vectors a (scale_vector (-3) b)) ↔ k = -3 := by sorry

theorem parallel_condition (k : ℝ) : 
  parallel (add_vectors (scale_vector k a) b) (add_vectors a (scale_vector (-3) b)) ↔ k = -1/3 := by sorry

end perpendicular_condition_parallel_condition_l1831_183152


namespace floor_abs_sum_abs_floor_l1831_183114

theorem floor_abs_sum_abs_floor : ⌊|(-5.7:ℝ)|⌋ + |⌊(-5.7:ℝ)⌋| = 11 := by
  sorry

end floor_abs_sum_abs_floor_l1831_183114


namespace largest_whole_number_less_than_150_over_11_l1831_183118

theorem largest_whole_number_less_than_150_over_11 : 
  (∀ x : ℕ, x > 13 → 11 * x ≥ 150) ∧ (11 * 13 < 150) := by
  sorry

end largest_whole_number_less_than_150_over_11_l1831_183118


namespace statements_about_positive_numbers_l1831_183198

theorem statements_about_positive_numbers (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a > b → a^2 > b^2) ∧ 
  ((2 * a * b) / (a + b) < (a + b) / 2) ∧ 
  ((a^2 + b^2) / 2 > ((a + b) / 2)^2) := by
  sorry

end statements_about_positive_numbers_l1831_183198


namespace winner_received_55_percent_l1831_183149

/-- Represents an election with two candidates -/
structure Election where
  winner_votes : ℕ
  margin : ℕ

/-- Calculates the percentage of votes received by the winner -/
def winner_percentage (e : Election) : ℚ :=
  (e.winner_votes : ℚ) / ((e.winner_votes + (e.winner_votes - e.margin)) : ℚ) * 100

/-- Theorem stating that in the given election scenario, the winner received 55% of the votes -/
theorem winner_received_55_percent (e : Election) 
  (h1 : e.winner_votes = 550) 
  (h2 : e.margin = 100) : 
  winner_percentage e = 55 := by
  sorry

#eval winner_percentage ⟨550, 100⟩

end winner_received_55_percent_l1831_183149


namespace xy9z_divisible_by_132_l1831_183113

def is_form_xy9z (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ n = x * 1000 + y * 100 + 90 + z

def valid_numbers : Set ℕ := {3696, 4092, 6996, 7392}

theorem xy9z_divisible_by_132 :
  ∀ n : ℕ, is_form_xy9z n ∧ 132 ∣ n ↔ n ∈ valid_numbers := by sorry

end xy9z_divisible_by_132_l1831_183113


namespace convention_handshakes_theorem_l1831_183167

/-- The number of handshakes in a convention with multiple companies -/
def convention_handshakes (num_companies : ℕ) (representatives_per_company : ℕ) : ℕ :=
  let total_people := num_companies * representatives_per_company
  let handshakes_per_person := total_people - representatives_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a convention with 3 companies, each having 5 representatives,
    where each person shakes hands only once with every person except those
    from their own company, the total number of handshakes is 75. -/
theorem convention_handshakes_theorem :
  convention_handshakes 3 5 = 75 := by
  sorry

end convention_handshakes_theorem_l1831_183167


namespace race_head_start_l1831_183176

theorem race_head_start (vA vB L H : ℝ) : 
  vA = (15 / 13) * vB →
  (L - H) / vB = L / vA - 0.4 * L / vB →
  H = (8 / 15) * L :=
by sorry

end race_head_start_l1831_183176


namespace cost_of_shoes_l1831_183124

def monthly_allowance : ℕ := 5
def months_saved : ℕ := 3
def lawn_mowing_fee : ℕ := 15
def lawns_mowed : ℕ := 4
def driveway_shoveling_fee : ℕ := 7
def driveways_shoveled : ℕ := 5
def change_left : ℕ := 15

def total_saved : ℕ := 
  monthly_allowance * months_saved + 
  lawn_mowing_fee * lawns_mowed + 
  driveway_shoveling_fee * driveways_shoveled

theorem cost_of_shoes : 
  total_saved - change_left = 95 := by
  sorry

end cost_of_shoes_l1831_183124


namespace gift_cost_theorem_l1831_183199

/-- Calculates the total cost of gifts for all workers in a company -/
def total_gift_cost (workers_per_block : ℕ) (num_blocks : ℕ) (gift_worth : ℕ) : ℕ :=
  workers_per_block * num_blocks * gift_worth

/-- The total cost of gifts for all workers in the company is $6000 -/
theorem gift_cost_theorem :
  total_gift_cost 200 15 2 = 6000 := by
  sorry

end gift_cost_theorem_l1831_183199


namespace ellipse_properties_l1831_183115

/-- An ellipse with center at the origin, foci on the x-axis, and max/min distances to focus 3 and 1 -/
structure Ellipse where
  center : ℝ × ℝ := (0, 0)
  foci_on_x_axis : Bool
  max_dist : ℝ := 3
  min_dist : ℝ := 1

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- A line with equation y = x + m -/
structure Line where
  m : ℝ

/-- Predicate for line intersection with ellipse -/
def intersects (l : Line) (e : Ellipse) : Prop :=
  ∃ x y : ℝ, y = x + l.m ∧ standard_equation e x y

theorem ellipse_properties (e : Ellipse) :
  (∀ x y : ℝ, standard_equation e x y ↔ 
    x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ l : Line, intersects l e ↔ 
    -Real.sqrt 7 ≤ l.m ∧ l.m ≤ Real.sqrt 7) := by sorry

end ellipse_properties_l1831_183115


namespace cad_to_jpy_exchange_l1831_183135

/-- The exchange rate from Canadian dollars (CAD) to Japanese yen (JPY) -/
def exchange_rate (cad : ℚ) (jpy : ℚ) : Prop :=
  5000 / 60 = jpy / cad

/-- The rounded exchange rate for 1 CAD in JPY -/
def rounded_rate (rate : ℚ) : ℕ :=
  (rate + 1/2).floor.toNat

theorem cad_to_jpy_exchange :
  ∃ (rate : ℚ), exchange_rate 1 rate ∧ rounded_rate rate = 83 := by
  sorry

end cad_to_jpy_exchange_l1831_183135


namespace first_shipment_cost_l1831_183157

/-- Represents the cost of a clothing shipment -/
def shipment_cost (num_sweaters num_jackets : ℕ) (sweater_price jacket_price : ℚ) : ℚ :=
  num_sweaters * sweater_price + num_jackets * jacket_price

theorem first_shipment_cost (sweater_price jacket_price : ℚ) :
  shipment_cost 5 15 sweater_price jacket_price = 550 →
  shipment_cost 10 20 sweater_price jacket_price = 1100 := by
  sorry

end first_shipment_cost_l1831_183157


namespace circle_symmetric_about_center_circle_symmetric_about_diameter_circle_is_symmetrical_l1831_183197

/-- Definition of a circle in a plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- Definition of symmetry for a set about a point -/
def IsSymmetricAbout (S : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, p.1 = (x.1 + y.1) / 2 ∧ p.2 = (x.2 + y.2) / 2

/-- Theorem: Any circle is symmetric about its center -/
theorem circle_symmetric_about_center (center : ℝ × ℝ) (radius : ℝ) :
  IsSymmetricAbout (Circle center radius) center := by
  sorry

/-- Theorem: Any circle is symmetric about any of its diameters -/
theorem circle_symmetric_about_diameter (center : ℝ × ℝ) (radius : ℝ) (a b : ℝ × ℝ) 
  (ha : a ∈ Circle center radius) (hb : b ∈ Circle center radius)
  (hdiameter : (a.1 - b.1)^2 + (a.2 - b.2)^2 = 4 * radius^2) :
  IsSymmetricAbout (Circle center radius) ((a.1 + b.1) / 2, (a.2 + b.2) / 2) := by
  sorry

/-- Main theorem: Any circle is a symmetrical figure -/
theorem circle_is_symmetrical (center : ℝ × ℝ) (radius : ℝ) :
  ∃ p, IsSymmetricAbout (Circle center radius) p := by
  sorry

end circle_symmetric_about_center_circle_symmetric_about_diameter_circle_is_symmetrical_l1831_183197


namespace arccos_cos_eq_two_thirds_x_l1831_183147

theorem arccos_cos_eq_two_thirds_x (x : Real) :
  0 ≤ x ∧ x ≤ (3 * Real.pi / 2) →
  (Real.arccos (Real.cos x) = 2 * x / 3) ↔ (x = 0 ∨ x = 6 * Real.pi / 5 ∨ x = 12 * Real.pi / 5) :=
by sorry

end arccos_cos_eq_two_thirds_x_l1831_183147


namespace pool_filling_time_l1831_183131

/-- Proves that filling a pool of 15,000 gallons with four hoses (two at 2 gal/min, two at 3 gal/min) takes 25 hours -/
theorem pool_filling_time : 
  let pool_volume : ℝ := 15000
  let hose_rate_1 : ℝ := 2
  let hose_rate_2 : ℝ := 3
  let num_hoses_1 : ℕ := 2
  let num_hoses_2 : ℕ := 2
  let total_rate : ℝ := hose_rate_1 * num_hoses_1 + hose_rate_2 * num_hoses_2
  let fill_time_minutes : ℝ := pool_volume / total_rate
  let fill_time_hours : ℝ := fill_time_minutes / 60
  fill_time_hours = 25 := by
sorry


end pool_filling_time_l1831_183131


namespace sum_of_solutions_quadratic_l1831_183156

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (72 - 18*x - x^2 = 0) → (∃ r s : ℝ, (72 - 18*r - r^2 = 0) ∧ (72 - 18*s - s^2 = 0) ∧ (r + s = 18)) :=
by sorry

end sum_of_solutions_quadratic_l1831_183156


namespace winnie_lollipops_left_l1831_183169

/-- The number of lollipops Winnie has left after distributing them equally among her friends -/
def lollipops_left (cherry wintergreen grape shrimp friends : ℕ) : ℕ :=
  (cherry + wintergreen + grape + shrimp) % friends

theorem winnie_lollipops_left :
  lollipops_left 32 150 7 280 14 = 7 := by
  sorry

end winnie_lollipops_left_l1831_183169


namespace expression_equality_l1831_183173

theorem expression_equality : 150 * (150 - 4) - (150 * 150 - 8 + 2^3) = -600 := by
  sorry

end expression_equality_l1831_183173


namespace expression_evaluation_l1831_183172

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end expression_evaluation_l1831_183172


namespace people_disliking_both_tv_and_games_l1831_183186

def total_surveyed : ℕ := 1500
def tv_dislike_percentage : ℚ := 25 / 100
def both_dislike_percentage : ℚ := 15 / 100

theorem people_disliking_both_tv_and_games :
  ⌊(tv_dislike_percentage * total_surveyed : ℚ) * both_dislike_percentage⌋ = 56 := by
  sorry

end people_disliking_both_tv_and_games_l1831_183186


namespace marble_ratio_l1831_183155

theorem marble_ratio : 
  let total_marbles : ℕ := 63
  let red_marbles : ℕ := 38
  let green_marbles : ℕ := 4
  let dark_blue_marbles : ℕ := total_marbles - red_marbles - green_marbles
  (dark_blue_marbles : ℚ) / total_marbles = 1 / 3 := by
  sorry

end marble_ratio_l1831_183155


namespace fertilizer_production_equation_l1831_183151

/-- Given a fertilizer factory with:
  * Original production plan of x tons per day
  * New production of x + 3 tons per day
  * Time to produce 180 tons (new rate) = Time to produce 120 tons (original rate)
  Prove that the equation 120/x = 180/(x + 3) correctly represents the relationship
  between the original production rate x and the time taken to produce different
  quantities of fertilizer. -/
theorem fertilizer_production_equation (x : ℝ) (h : x > 0) :
  (120 : ℝ) / x = 180 / (x + 3) ↔
  (120 : ℝ) / x = (180 : ℝ) / (x + 3) :=
by sorry

end fertilizer_production_equation_l1831_183151


namespace train_crossing_time_l1831_183188

/-- Time taken for a train to cross a man running in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 220 →
  train_speed = 80 * 1000 / 3600 →
  man_speed = 8 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 11 := by
  sorry

end train_crossing_time_l1831_183188


namespace P_symmetric_l1831_183123

/-- Definition of the polynomial sequence P_m -/
def P : ℕ → (ℚ → ℚ → ℚ → ℚ)
| 0 => λ _ _ _ => 1
| (m + 1) => λ x y z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

/-- Statement that P_m is symmetric for all m -/
theorem P_symmetric (m : ℕ) (x y z : ℚ) :
  P m x y z = P m y x z ∧
  P m x y z = P m x z y ∧
  P m x y z = P m y z x ∧
  P m x y z = P m z x y ∧
  P m x y z = P m z y x :=
by sorry

end P_symmetric_l1831_183123
