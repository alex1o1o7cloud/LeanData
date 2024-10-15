import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_integers_l1217_121725

theorem sum_of_integers (a b c d e : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + e = 9)
  (eq4 : d - e + a = 4)
  (eq5 : e - a + b = 3) : 
  a + b + c + d + e = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1217_121725


namespace NUMINAMATH_CALUDE_lcd_of_fractions_l1217_121710

theorem lcd_of_fractions (a b c d : ℕ) (ha : a = 2) (hb : b = 4) (hc : c = 5) (hd : d = 6) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcd_of_fractions_l1217_121710


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l1217_121747

theorem sandy_shopping_money (remaining_amount : ℝ) (spent_percentage : ℝ) (initial_amount : ℝ) :
  remaining_amount = 224 ∧
  spent_percentage = 0.3 ∧
  remaining_amount = initial_amount * (1 - spent_percentage) →
  initial_amount = 320 :=
by sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l1217_121747


namespace NUMINAMATH_CALUDE_line_points_property_l1217_121785

theorem line_points_property (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -2 * x₁ + 3)
  (h2 : y₂ = -2 * x₂ + 3)
  (h3 : y₃ = -2 * x₃ + 3)
  (h4 : x₁ < x₂)
  (h5 : x₂ < x₃)
  (h6 : x₂ * x₃ < 0) :
  y₁ * y₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_points_property_l1217_121785


namespace NUMINAMATH_CALUDE_vector_coordinates_l1217_121769

/-- Given a vector a with magnitude √5 that is parallel to vector b=(1,2),
    prove that the coordinates of a are either (1,2) or (-1,-2) -/
theorem vector_coordinates (a b : ℝ × ℝ) : 
  (‖a‖ = Real.sqrt 5) → 
  (b = (1, 2)) → 
  (∃ (k : ℝ), a = k • b) → 
  (a = (1, 2) ∨ a = (-1, -2)) := by
  sorry

#check vector_coordinates

end NUMINAMATH_CALUDE_vector_coordinates_l1217_121769


namespace NUMINAMATH_CALUDE_biased_coin_theorem_l1217_121706

/-- The probability of getting heads in one flip of a biased coin -/
def h : ℚ := 3/7

/-- The probability of getting exactly k heads in n flips -/
def prob_k_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1-p)^(n-k)

theorem biased_coin_theorem :
  (prob_k_heads 6 2 h ≠ 0) ∧
  (prob_k_heads 6 2 h = prob_k_heads 6 3 h) ∧
  (h = 3/7) ∧
  (prob_k_heads 6 4 h = 240/1453) := by
  sorry

#eval Nat.gcd 240 1453 -- To verify that 240/1453 is in lowest terms

#eval 240 + 1453 -- To verify the final answer

end NUMINAMATH_CALUDE_biased_coin_theorem_l1217_121706


namespace NUMINAMATH_CALUDE_additional_money_needed_l1217_121731

/-- The cost of the dictionary -/
def dictionary_cost : ℚ := 5.50

/-- The cost of the dinosaur book -/
def dinosaur_book_cost : ℚ := 11.25

/-- The cost of the children's cookbook -/
def cookbook_cost : ℚ := 5.75

/-- The cost of the science experiment kit -/
def science_kit_cost : ℚ := 8.40

/-- The cost of the set of colored pencils -/
def pencils_cost : ℚ := 3.60

/-- The amount Emir has saved -/
def saved_amount : ℚ := 24.50

/-- The theorem stating how much more money Emir needs -/
theorem additional_money_needed :
  dictionary_cost + dinosaur_book_cost + cookbook_cost + science_kit_cost + pencils_cost - saved_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l1217_121731


namespace NUMINAMATH_CALUDE_distinct_bracelets_count_l1217_121772

/-- Represents a bead color -/
inductive BeadColor
| Red
| Blue
| Purple

/-- Represents a bracelet as a circular arrangement of beads -/
def Bracelet := List BeadColor

/-- Checks if two bracelets are equivalent under rotation and reflection -/
def are_equivalent (b1 b2 : Bracelet) : Bool :=
  sorry

/-- Counts the number of beads of each color in a bracelet -/
def count_beads (b : Bracelet) : Nat × Nat × Nat :=
  sorry

/-- Generates all possible bracelets with 2 red, 2 blue, and 2 purple beads -/
def generate_bracelets : List Bracelet :=
  sorry

/-- Counts the number of distinct bracelets -/
def count_distinct_bracelets : Nat :=
  sorry

/-- Theorem: The number of distinct bracelets with 2 red, 2 blue, and 2 purple beads is 11 -/
theorem distinct_bracelets_count :
  count_distinct_bracelets = 11 := by
  sorry

end NUMINAMATH_CALUDE_distinct_bracelets_count_l1217_121772


namespace NUMINAMATH_CALUDE_cricket_team_ratio_proof_l1217_121703

def cricket_team_ratio (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) : Prop :=
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed_non_throwers := non_throwers - right_handed_non_throwers
  2 * left_handed_non_throwers = right_handed_non_throwers

theorem cricket_team_ratio_proof :
  cricket_team_ratio 64 37 55 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_ratio_proof_l1217_121703


namespace NUMINAMATH_CALUDE_multiply_subtract_equal_compute_expression_l1217_121782

theorem multiply_subtract_equal (a b c : ℤ) : a * c - b * c = (a - b) * c := by sorry

theorem compute_expression : 45 * 1313 - 10 * 1313 = 45955 := by sorry

end NUMINAMATH_CALUDE_multiply_subtract_equal_compute_expression_l1217_121782


namespace NUMINAMATH_CALUDE_balance_after_transfer_l1217_121757

def initial_balance : ℝ := 400
def transfer_amount : ℝ := 90
def service_charge_rate : ℝ := 0.02

def final_balance : ℝ := initial_balance - (transfer_amount * (1 + service_charge_rate))

theorem balance_after_transfer :
  final_balance = 308.2 := by sorry

end NUMINAMATH_CALUDE_balance_after_transfer_l1217_121757


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l1217_121771

theorem binomial_expansion_problem (x y : ℝ) (n : ℕ) 
  (h1 : n * x^(n-1) * y = 240)
  (h2 : n * (n-1) / 2 * x^(n-2) * y^2 = 720)
  (h3 : n * (n-1) * (n-2) / 6 * x^(n-3) * y^3 = 1080) :
  x = 2 ∧ y = 3 ∧ n = 5 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l1217_121771


namespace NUMINAMATH_CALUDE_complement_intersection_l1217_121762

theorem complement_intersection (I A B : Set ℕ) : 
  I = {1, 2, 3, 4, 5} →
  A = {1, 2} →
  B = {1, 3, 5} →
  (I \ A) ∩ B = {3, 5} := by
sorry

end NUMINAMATH_CALUDE_complement_intersection_l1217_121762


namespace NUMINAMATH_CALUDE_triangle_inequality_last_three_terms_l1217_121775

/-- An arithmetic sequence with positive terms and positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n + d

/-- Triangle inequality for the last three terms of a four-term arithmetic sequence -/
theorem triangle_inequality_last_three_terms
  (a : ℕ → ℝ) (d : ℝ) (h : ArithmeticSequence a d) :
  a 2 + a 3 > a 4 ∧ a 2 + a 4 > a 3 ∧ a 3 + a 4 > a 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_last_three_terms_l1217_121775


namespace NUMINAMATH_CALUDE_angle_bisector_product_theorem_l1217_121789

/-- Given a triangle with sides a, b, c, internal angle bisectors fa, fb, fc, and area T,
    this theorem states that the product of the angle bisectors divided by the product of the sides
    is equal to four times the area multiplied by the sum of the sides,
    divided by the product of the pairwise sums of the sides. -/
theorem angle_bisector_product_theorem
  (a b c fa fb fc T : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b)
  (h_bisectors : fa > 0 ∧ fb > 0 ∧ fc > 0)
  (h_area : T > 0) :
  (fa * fb * fc) / (a * b * c) = 4 * T * (a + b + c) / ((a + b) * (b + c) * (a + c)) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_product_theorem_l1217_121789


namespace NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l1217_121780

noncomputable def f (x : ℝ) := x / Real.exp x

theorem f_monotonicity_and_inequality :
  (∀ x y, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) ∧
  (∀ x, x > 0 → Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x)) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l1217_121780


namespace NUMINAMATH_CALUDE_right_triangle_legs_sum_l1217_121767

theorem right_triangle_legs_sum (a b c : ℕ) : 
  a + 1 = b →                -- legs are consecutive integers
  a^2 + b^2 = 41^2 →         -- Pythagorean theorem with hypotenuse 41
  a + b = 59 :=              -- sum of legs is 59
by sorry

end NUMINAMATH_CALUDE_right_triangle_legs_sum_l1217_121767


namespace NUMINAMATH_CALUDE_max_students_distribution_l1217_121714

theorem max_students_distribution (pens pencils : ℕ) 
  (h_pens : pens = 1230) (h_pencils : pencils = 920) : 
  (Nat.gcd pens pencils) = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l1217_121714


namespace NUMINAMATH_CALUDE_age_difference_l1217_121779

/-- Given that the sum of X and Y is 12 years greater than the sum of Y and Z,
    prove that Z is 12 years younger than X. -/
theorem age_difference (X Y Z : ℕ) (h : X + Y = Y + Z + 12) : X - Z = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1217_121779


namespace NUMINAMATH_CALUDE_green_toads_count_l1217_121793

/-- The number of green toads per acre -/
def green_toads_per_acre : ℕ := 8

/-- The ratio of green toads to brown toads -/
def green_to_brown_ratio : ℚ := 1 / 25

/-- The fraction of brown toads that are spotted -/
def spotted_brown_fraction : ℚ := 1 / 4

/-- The number of spotted brown toads per acre -/
def spotted_brown_per_acre : ℕ := 50

theorem green_toads_count :
  green_toads_per_acre = 8 :=
sorry

end NUMINAMATH_CALUDE_green_toads_count_l1217_121793


namespace NUMINAMATH_CALUDE_sirokas_guests_l1217_121722

/-- The number of guests Mrs. Široká was expecting -/
def num_guests : ℕ := 11

/-- The number of sandwiches in the first scenario -/
def sandwiches1 : ℕ := 25

/-- The number of sandwiches in the second scenario -/
def sandwiches2 : ℕ := 35

/-- The number of sandwiches in the final scenario -/
def sandwiches3 : ℕ := 52

theorem sirokas_guests :
  (sandwiches1 < 2 * num_guests + 3) ∧
  (sandwiches1 ≥ 2 * num_guests) ∧
  (sandwiches2 < 3 * num_guests + 4) ∧
  (sandwiches2 ≥ 3 * num_guests) ∧
  (sandwiches3 ≥ 4 * num_guests) ∧
  (sandwiches3 < 5 * num_guests) :=
by sorry

end NUMINAMATH_CALUDE_sirokas_guests_l1217_121722


namespace NUMINAMATH_CALUDE_angle_XYZ_measure_l1217_121729

-- Define the regular octagon
def RegularOctagon : Type := Unit

-- Define the square inside the octagon
def Square : Type := Unit

-- Define the vertices
def X : RegularOctagon := Unit.unit
def Y : Square := Unit.unit
def Z : Square := Unit.unit

-- Define the angle measure function
def angle_measure : RegularOctagon → Square → Square → ℝ := sorry

-- State the theorem
theorem angle_XYZ_measure (o : RegularOctagon) (s : Square) :
  angle_measure X Y Z = 90 := by sorry

end NUMINAMATH_CALUDE_angle_XYZ_measure_l1217_121729


namespace NUMINAMATH_CALUDE_A_intersect_C_U_B_eq_open_zero_closed_two_l1217_121750

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | x > 2}

-- Define the complement of B in the universal set ℝ
def C_U_B : Set ℝ := {x | ¬ (x ∈ B)}

-- State the theorem
theorem A_intersect_C_U_B_eq_open_zero_closed_two : 
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_C_U_B_eq_open_zero_closed_two_l1217_121750


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1217_121732

theorem trigonometric_identities :
  (Real.sin (20 * π / 180))^2 + (Real.cos (80 * π / 180))^2 + Real.sqrt 3 * Real.sin (20 * π / 180) * Real.cos (80 * π / 180) = 1/4 ∧
  (Real.sin (20 * π / 180))^2 + (Real.cos (50 * π / 180))^2 + Real.sin (20 * π / 180) * Real.cos (50 * π / 180) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1217_121732


namespace NUMINAMATH_CALUDE_cristina_catches_nicky_l1217_121753

/-- Proves that Cristina catches up to Nicky in 27 seconds --/
theorem cristina_catches_nicky (cristina_speed nicky_speed : ℝ) (head_start : ℝ) 
  (h1 : cristina_speed > nicky_speed)
  (h2 : cristina_speed = 5)
  (h3 : nicky_speed = 3)
  (h4 : head_start = 54) :
  (head_start / (cristina_speed - nicky_speed) = 27) := by
  sorry

end NUMINAMATH_CALUDE_cristina_catches_nicky_l1217_121753


namespace NUMINAMATH_CALUDE_initial_number_of_boys_l1217_121707

theorem initial_number_of_boys (B : ℝ) : 
  (1.2 * B) + B + (2.4 * B) = 51 → B = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_boys_l1217_121707


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l1217_121705

/-- The set of points (x, y) satisfying the given equation forms an ellipse -/
theorem trajectory_is_ellipse :
  ∀ x y : ℝ, 
  Real.sqrt (x^2 + (y + 3)^2) + Real.sqrt (x^2 + (y - 3)^2) = 10 →
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
  x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l1217_121705


namespace NUMINAMATH_CALUDE_bakery_flour_usage_l1217_121749

theorem bakery_flour_usage (wheat_flour : ℝ) (white_flour : ℝ) 
  (h1 : wheat_flour = 0.2)
  (h2 : white_flour = 0.1) :
  wheat_flour + white_flour = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_usage_l1217_121749


namespace NUMINAMATH_CALUDE_inequality_proof_l1217_121721

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1217_121721


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l1217_121702

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  ∃ (k : ℤ), (15 * x + 3) * (15 * x + 9) * (5 * x + 10) = 90 * k ∧
  ∀ (m : ℤ), m > 90 → ¬(∀ (y : ℤ), Even y →
    ∃ (l : ℤ), (15 * y + 3) * (15 * y + 9) * (5 * y + 10) = m * l) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l1217_121702


namespace NUMINAMATH_CALUDE_squared_ratios_sum_ge_sum_l1217_121748

theorem squared_ratios_sum_ge_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ a + b + c := by sorry

end NUMINAMATH_CALUDE_squared_ratios_sum_ge_sum_l1217_121748


namespace NUMINAMATH_CALUDE_cat_video_length_is_correct_l1217_121735

/-- The length of the cat video in minutes -/
def cat_video_length : ℝ := 4

/-- The total time spent watching videos in minutes -/
def total_watching_time : ℝ := 36

/-- Theorem stating that the cat video length is correct given the conditions -/
theorem cat_video_length_is_correct :
  let dog_video_length := 2 * cat_video_length
  let gorilla_video_length := 2 * (cat_video_length + dog_video_length)
  cat_video_length + dog_video_length + gorilla_video_length = total_watching_time :=
by sorry

end NUMINAMATH_CALUDE_cat_video_length_is_correct_l1217_121735


namespace NUMINAMATH_CALUDE_vartan_recreation_spending_l1217_121766

theorem vartan_recreation_spending :
  ∀ (last_week_wages : ℝ) (last_week_percent : ℝ),
  last_week_percent > 0 →
  let this_week_wages := 0.9 * last_week_wages
  let last_week_spending := (last_week_percent / 100) * last_week_wages
  let this_week_spending := 0.3 * this_week_wages
  this_week_spending = 1.8 * last_week_spending →
  last_week_percent = 15 := by
sorry

end NUMINAMATH_CALUDE_vartan_recreation_spending_l1217_121766


namespace NUMINAMATH_CALUDE_contact_box_price_l1217_121701

/-- The price of a box of contacts given the number of contacts and cost per contact -/
def box_price (num_contacts : ℕ) (cost_per_contact : ℚ) : ℚ :=
  num_contacts * cost_per_contact

/-- The cost per contact for a box given its total price and number of contacts -/
def cost_per_contact (total_price : ℚ) (num_contacts : ℕ) : ℚ :=
  total_price / num_contacts

theorem contact_box_price :
  let box1_contacts : ℕ := 50
  let box2_contacts : ℕ := 99
  let box2_price : ℚ := 33

  let box2_cost_per_contact := cost_per_contact box2_price box2_contacts
  let chosen_cost_per_contact : ℚ := 1 / 3

  box_price box1_contacts chosen_cost_per_contact = 50 * (1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_contact_box_price_l1217_121701


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1217_121777

def triangle_area (r R : ℝ) (cosA cosB cosC : ℝ) (a b c : ℝ) : Prop :=
  r = 7 ∧
  R = 20 ∧
  3 * cosB = 2 * cosA + cosC ∧
  cosA + cosB + cosC = 1 + r / R ∧
  b = 2 * R * Real.sqrt (1 - cosB^2) ∧
  a^2 + c^2 - a * c * cosB = b^2 ∧
  cosA = (b^2 + c^2 - a^2) / (2 * b * c) ∧
  cosC = (a^2 + b^2 - c^2) / (2 * a * b) ∧
  (7 * (a + c + 2 * Real.sqrt 319)) / 2 = 7 * ((a + b + c) / 2)

theorem triangle_area_theorem :
  ∀ (r R : ℝ) (cosA cosB cosC : ℝ) (a b c : ℝ),
    triangle_area r R cosA cosB cosC a b c →
    (7 * (a + c + 2 * Real.sqrt 319)) / 2 = 7 * ((a + b + c) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1217_121777


namespace NUMINAMATH_CALUDE_a_composition_zero_l1217_121700

def a (k : ℕ) : ℕ := (2 * k + 1) ^ k

theorem a_composition_zero : a (a (a 0)) = 343 := by sorry

end NUMINAMATH_CALUDE_a_composition_zero_l1217_121700


namespace NUMINAMATH_CALUDE_cube_frame_problem_solution_l1217_121723

/-- Represents the cube frame construction problem. -/
structure CubeFrameProblem where
  bonnie_wire_length : ℕ        -- Length of each wire piece Bonnie uses
  bonnie_wire_count : ℕ         -- Number of wire pieces Bonnie uses
  roark_wire_length : ℕ         -- Length of each wire piece Roark uses
  roark_cube_edge_length : ℕ    -- Edge length of Roark's unit cubes

/-- The solution to the cube frame problem. -/
def cubeProblemSolution (p : CubeFrameProblem) : ℚ :=
  let bonnie_total_length := p.bonnie_wire_length * p.bonnie_wire_count
  let bonnie_cube_volume := p.bonnie_wire_length ^ 3
  let roark_cube_count := bonnie_cube_volume
  let roark_wire_per_cube := 12 * p.roark_wire_length
  let roark_total_length := roark_cube_count * roark_wire_per_cube
  bonnie_total_length / roark_total_length

/-- Theorem stating the solution to the cube frame problem. -/
theorem cube_frame_problem_solution (p : CubeFrameProblem) 
  (h1 : p.bonnie_wire_length = 8)
  (h2 : p.bonnie_wire_count = 12)
  (h3 : p.roark_wire_length = 2)
  (h4 : p.roark_cube_edge_length = 1) :
  cubeProblemSolution p = 1 / 128 := by
  sorry

end NUMINAMATH_CALUDE_cube_frame_problem_solution_l1217_121723


namespace NUMINAMATH_CALUDE_wheat_packets_fill_gunny_bag_l1217_121760

/-- The number of pounds in one ton -/
def pounds_per_ton : ℕ := 2300

/-- The number of packets of wheat -/
def num_packets : ℕ := 1840

/-- The weight of each packet in pounds -/
def packet_weight_pounds : ℕ := 16

/-- The additional weight of each packet in ounces -/
def packet_weight_ounces : ℕ := 4

/-- The capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℕ := 13

/-- The number of ounces in one pound -/
def ounces_per_pound : ℕ := 16

theorem wheat_packets_fill_gunny_bag :
  ounces_per_pound = 16 :=
sorry

end NUMINAMATH_CALUDE_wheat_packets_fill_gunny_bag_l1217_121760


namespace NUMINAMATH_CALUDE_floor_of_5_7_l1217_121768

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_5_7_l1217_121768


namespace NUMINAMATH_CALUDE_elections_with_at_least_two_past_officers_l1217_121794

def total_candidates : ℕ := 20
def past_officers : ℕ := 10
def positions : ℕ := 6

def total_elections : ℕ := Nat.choose total_candidates positions

def elections_no_past_officers : ℕ := Nat.choose (total_candidates - past_officers) positions

def elections_one_past_officer : ℕ := 
  Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions - 1)

theorem elections_with_at_least_two_past_officers : 
  total_elections - elections_no_past_officers - elections_one_past_officer = 36030 := by
  sorry

end NUMINAMATH_CALUDE_elections_with_at_least_two_past_officers_l1217_121794


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1217_121755

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x = 3 → x^2 = 9) ∧ 
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1217_121755


namespace NUMINAMATH_CALUDE_intersection_points_parallel_lines_l1217_121745

/-- Given two parallel lines with m and n points respectively, 
    this theorem states the number of intersection points formed by 
    segments connecting these points. -/
theorem intersection_points_parallel_lines 
  (m n : ℕ) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_parallel_lines_l1217_121745


namespace NUMINAMATH_CALUDE_order_xyz_l1217_121751

theorem order_xyz (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (x : ℝ) (hx : x = (a+b)*(c+d))
  (y : ℝ) (hy : y = (a+c)*(b+d))
  (z : ℝ) (hz : z = (a+d)*(b+c)) :
  x < y ∧ y < z :=
by sorry

end NUMINAMATH_CALUDE_order_xyz_l1217_121751


namespace NUMINAMATH_CALUDE_correct_atomic_symbol_proof_l1217_121719

/-- Represents an element X in an ionic compound XCl_n -/
structure ElementX where
  m : ℕ  -- number of neutrons
  y : ℕ  -- number of electrons outside the nucleus
  n : ℕ  -- number of chlorine atoms in the compound

/-- Represents the atomic symbol of an isotope -/
structure AtomicSymbol where
  subscript : ℕ
  superscript : ℕ

/-- Returns the correct atomic symbol for an element X -/
def correct_atomic_symbol (x : ElementX) : AtomicSymbol :=
  { subscript := x.y + x.n
  , superscript := x.m + x.y + x.n }

/-- Theorem stating that the correct atomic symbol for element X is _{y+n}^{m+y+n}X -/
theorem correct_atomic_symbol_proof (x : ElementX) :
  correct_atomic_symbol x = { subscript := x.y + x.n, superscript := x.m + x.y + x.n } :=
by sorry

end NUMINAMATH_CALUDE_correct_atomic_symbol_proof_l1217_121719


namespace NUMINAMATH_CALUDE_odd_numbers_pascal_triangle_l1217_121704

/-- 
Given a non-negative integer n, count_ones n returns the number of 1's 
in the binary representation of n.
-/
def count_ones (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 2) + count_ones (n / 2)

/-- 
Given a non-negative integer n, odd_numbers_in_pascal_row n returns the 
number of odd numbers in the n-th row of Pascal's triangle.
-/
def odd_numbers_in_pascal_row (n : ℕ) : ℕ :=
  2^(count_ones n)

/-- 
Theorem: The number of odd numbers in the n-th row of Pascal's triangle 
is equal to 2^k, where k is the number of 1's in the binary representation of n.
-/
theorem odd_numbers_pascal_triangle (n : ℕ) : 
  odd_numbers_in_pascal_row n = 2^(count_ones n) := by
  sorry


end NUMINAMATH_CALUDE_odd_numbers_pascal_triangle_l1217_121704


namespace NUMINAMATH_CALUDE_trig_identity_l1217_121781

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos (2 * α) + Real.sin (2 * α)) = 5 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l1217_121781


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1217_121790

theorem perfect_square_trinomial 
  (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^2) :
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1217_121790


namespace NUMINAMATH_CALUDE_correct_calculation_l1217_121736

theorem correct_calculation (x : ℝ) (h : 7 * x = 70) : 36 - x = 26 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1217_121736


namespace NUMINAMATH_CALUDE_eiffel_tower_height_is_324_l1217_121799

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height : ℝ := 830

/-- The difference in height between the Burj Khalifa and the Eiffel Tower in meters -/
def height_difference : ℝ := 506

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height : ℝ := burj_khalifa_height - height_difference

/-- Proves that the height of the Eiffel Tower is 324 meters -/
theorem eiffel_tower_height_is_324 : eiffel_tower_height = 324 := by
  sorry

end NUMINAMATH_CALUDE_eiffel_tower_height_is_324_l1217_121799


namespace NUMINAMATH_CALUDE_proposition_relationship_l1217_121770

theorem proposition_relationship :
  ∀ (p q : Prop),
  (p → q) →                        -- Proposition A: p is sufficient for q
  (p ↔ q) →                        -- Proposition B: p is necessary and sufficient for q
  ((p ↔ q) → (p → q)) ∧            -- Proposition A is necessary for Proposition B
  ¬((p → q) → (p ↔ q)) :=          -- Proposition A is not sufficient for Proposition B
by
  sorry

end NUMINAMATH_CALUDE_proposition_relationship_l1217_121770


namespace NUMINAMATH_CALUDE_inequality_proof_l1217_121715

theorem inequality_proof (k n : ℕ) (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) :
  (1 - (1 - x)^n)^k ≥ 1 - (1 - x^k)^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1217_121715


namespace NUMINAMATH_CALUDE_range_of_expression_l1217_121708

theorem range_of_expression (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -3/2 < 1/2 * α - β ∧ 1/2 * α - β < 11/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l1217_121708


namespace NUMINAMATH_CALUDE_only_set_A_forms_triangle_l1217_121791

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem only_set_A_forms_triangle :
  can_form_triangle 3 4 5 ∧
  ¬can_form_triangle 4 4 8 ∧
  ¬can_form_triangle 3 10 4 ∧
  ¬can_form_triangle 4 5 10 :=
sorry

end NUMINAMATH_CALUDE_only_set_A_forms_triangle_l1217_121791


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l1217_121774

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l1217_121774


namespace NUMINAMATH_CALUDE_jackson_running_program_l1217_121786

/-- Calculates the final running distance after a given number of days,
    given an initial distance and daily increase. -/
def finalRunningDistance (initialDistance : ℝ) (dailyIncrease : ℝ) (days : ℕ) : ℝ :=
  initialDistance + dailyIncrease * (days - 1)

/-- Theorem stating that given the initial conditions of Jackson's running program,
    the final running distance on the last day is 16.5 miles. -/
theorem jackson_running_program :
  let initialDistance : ℝ := 3
  let dailyIncrease : ℝ := 0.5
  let programDays : ℕ := 28
  finalRunningDistance initialDistance dailyIncrease programDays = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_jackson_running_program_l1217_121786


namespace NUMINAMATH_CALUDE_library_book_return_percentage_l1217_121740

theorem library_book_return_percentage 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (loaned_books : ℕ) 
  (h1 : initial_books = 300) 
  (h2 : final_books = 244) 
  (h3 : loaned_books = 160) : 
  (((loaned_books - (initial_books - final_books)) / loaned_books) * 100 : ℚ) = 65 := by
  sorry

end NUMINAMATH_CALUDE_library_book_return_percentage_l1217_121740


namespace NUMINAMATH_CALUDE_unique_solution_exists_l1217_121724

theorem unique_solution_exists : 
  ∃! (a b c : ℕ+), 
    (a.val * b.val + 3 * b.val * c.val = 63) ∧ 
    (a.val * c.val + 3 * b.val * c.val = 39) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l1217_121724


namespace NUMINAMATH_CALUDE_fourth_power_inequality_l1217_121778

theorem fourth_power_inequality (a b c : ℝ) :
  a^4 + b^4 + c^4 ≥ a*b*c*(a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_inequality_l1217_121778


namespace NUMINAMATH_CALUDE_planes_through_three_points_l1217_121712

/-- Three points in 3D space -/
structure ThreePoints where
  p1 : ℝ × ℝ × ℝ
  p2 : ℝ × ℝ × ℝ
  p3 : ℝ × ℝ × ℝ

/-- Possible number of planes through three points -/
inductive NumPlanes
  | one
  | infinite

/-- The number of planes that can be constructed through three points in 3D space 
    is either one or infinite -/
theorem planes_through_three_points (points : ThreePoints) : 
  ∃ (n : NumPlanes), n = NumPlanes.one ∨ n = NumPlanes.infinite :=
sorry

end NUMINAMATH_CALUDE_planes_through_three_points_l1217_121712


namespace NUMINAMATH_CALUDE_email_count_correct_l1217_121713

/-- Calculates the number of emails in Jackson's inbox after deletion and reception process -/
def final_email_count (deleted1 deleted2 received1 received2 received_after : ℕ) : ℕ :=
  received1 + received2 + received_after

/-- Theorem stating that the final email count is correct given the problem conditions -/
theorem email_count_correct :
  let deleted1 := 50
  let deleted2 := 20
  let received1 := 15
  let received2 := 5
  let received_after := 10
  final_email_count deleted1 deleted2 received1 received2 received_after = 30 := by sorry

end NUMINAMATH_CALUDE_email_count_correct_l1217_121713


namespace NUMINAMATH_CALUDE_seashells_given_to_joan_l1217_121726

theorem seashells_given_to_joan (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 8) 
  (h2 : remaining_seashells = 2) : 
  initial_seashells - remaining_seashells = 6 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_joan_l1217_121726


namespace NUMINAMATH_CALUDE_robin_oatmeal_cookies_l1217_121764

/-- Calculates the number of oatmeal cookies Robin had -/
def oatmeal_cookies (cookies_per_bag : ℕ) (chocolate_chip_cookies : ℕ) (baggies : ℕ) : ℕ :=
  cookies_per_bag * baggies - chocolate_chip_cookies

/-- Proves that Robin had 25 oatmeal cookies -/
theorem robin_oatmeal_cookies :
  oatmeal_cookies 6 23 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_robin_oatmeal_cookies_l1217_121764


namespace NUMINAMATH_CALUDE_science_club_problem_l1217_121738

theorem science_club_problem (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : math = 85)
  (h3 : physics = 60)
  (h4 : both = 20) :
  total - (math + physics - both) = 25 := by
sorry

end NUMINAMATH_CALUDE_science_club_problem_l1217_121738


namespace NUMINAMATH_CALUDE_rows_of_nine_l1217_121763

/-- Given 74 people seated in rows of either 7 or 9 seats, with all seats occupied,
    there are exactly 2 rows seating 9 people. -/
theorem rows_of_nine (total_people : ℕ) (rows_of_seven : ℕ) (rows_of_nine : ℕ) : 
  total_people = 74 →
  total_people = 7 * rows_of_seven + 9 * rows_of_nine →
  rows_of_nine = 2 := by
  sorry

end NUMINAMATH_CALUDE_rows_of_nine_l1217_121763


namespace NUMINAMATH_CALUDE_simplify_expression_l1217_121743

theorem simplify_expression : 
  2 * Real.sqrt 12 + 3 * Real.sqrt (4/3) - Real.sqrt (16/3) - 2/3 * Real.sqrt 48 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1217_121743


namespace NUMINAMATH_CALUDE_max_log_sum_l1217_121783

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = 40) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 4*b = 40 → Real.log a + Real.log b ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_log_sum_l1217_121783


namespace NUMINAMATH_CALUDE_value_subtracted_l1217_121752

theorem value_subtracted (n : ℝ) (x : ℝ) : 
  (2 * n + 20 = 8 * n - x) → 
  (n = 4) → 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_value_subtracted_l1217_121752


namespace NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l1217_121773

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem f_geq_a_iff_a_in_range (a : ℝ) :
  (∀ x ≥ (1/2 : ℝ), f a x ≥ a) ↔ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l1217_121773


namespace NUMINAMATH_CALUDE_outbound_speed_calculation_l1217_121798

-- Define the problem parameters
def distance : ℝ := 19.999999999999996
def return_speed : ℝ := 4
def total_time : ℝ := 5.8

-- Define the theorem
theorem outbound_speed_calculation :
  ∃ (v : ℝ), v > 0 ∧ (distance / v + distance / return_speed = total_time) → v = 25 := by
  sorry

end NUMINAMATH_CALUDE_outbound_speed_calculation_l1217_121798


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l1217_121765

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to its decimal representation -/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- The sum of digits of a TwoDigitNumber -/
def TwoDigitNumber.digitSum (n : TwoDigitNumber) : Nat :=
  n.tens + n.ones

/-- The product of digits of a TwoDigitNumber -/
def TwoDigitNumber.digitProduct (n : TwoDigitNumber) : Nat :=
  n.tens * n.ones

theorem unique_two_digit_number :
  ∃! (n : TwoDigitNumber),
    (n.toNat / n.digitSum = 4 ∧ n.toNat % n.digitSum = 3) ∧
    (n.toNat / n.digitProduct = 3 ∧ n.toNat % n.digitProduct = 5) ∧
    n.toNat = 23 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l1217_121765


namespace NUMINAMATH_CALUDE_honey_jars_needed_l1217_121792

theorem honey_jars_needed (num_hives : ℕ) (honey_per_hive : ℝ) (jar_capacity : ℝ) 
  (h1 : num_hives = 5)
  (h2 : honey_per_hive = 20)
  (h3 : jar_capacity = 0.5)
  (h4 : jar_capacity > 0) :
  ⌈(↑num_hives * honey_per_hive / 2) / jar_capacity⌉ = 100 := by
  sorry

end NUMINAMATH_CALUDE_honey_jars_needed_l1217_121792


namespace NUMINAMATH_CALUDE_average_marker_cost_correct_l1217_121758

def average_marker_cost (num_markers : ℕ) (marker_price : ℚ) (handling_fee : ℚ) (shipping_cost : ℚ) : ℕ :=
  let total_cost := marker_price + (num_markers : ℚ) * handling_fee + shipping_cost
  let total_cents := (total_cost * 100).floor
  let average_cents := (total_cents + (num_markers / 2)) / num_markers
  average_cents.toNat

theorem average_marker_cost_correct :
  average_marker_cost 300 45 0.1 8.5 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_marker_cost_correct_l1217_121758


namespace NUMINAMATH_CALUDE_range_of_a_l1217_121744

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- State the theorem
theorem range_of_a (a : ℝ) (h : a ≠ 0) :
  C a ⊆ (A ∩ (Set.univ \ B)) →
  (0 < a ∧ a ≤ 2/3) ∨ (-2/3 ≤ a ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1217_121744


namespace NUMINAMATH_CALUDE_square_area_from_string_length_l1217_121727

theorem square_area_from_string_length (string_length : ℝ) (h : string_length = 32) :
  let side_length := string_length / 4
  side_length * side_length = 64 := by sorry

end NUMINAMATH_CALUDE_square_area_from_string_length_l1217_121727


namespace NUMINAMATH_CALUDE_minimum_bottles_needed_l1217_121717

/-- The capacity of a medium-sized bottle in milliliters -/
def medium_bottle_capacity : ℕ := 80

/-- The capacity of a very large bottle in milliliters -/
def large_bottle_capacity : ℕ := 1200

/-- The maximum number of additional bottles allowed -/
def max_additional_bottles : ℕ := 5

/-- The minimum number of medium-sized bottles needed -/
def min_bottles_needed : ℕ := 15

theorem minimum_bottles_needed :
  (large_bottle_capacity / medium_bottle_capacity = min_bottles_needed) ∧
  (min_bottles_needed + max_additional_bottles ≥ 
   (large_bottle_capacity + medium_bottle_capacity - 1) / medium_bottle_capacity) :=
sorry

end NUMINAMATH_CALUDE_minimum_bottles_needed_l1217_121717


namespace NUMINAMATH_CALUDE_lap_length_l1217_121728

/-- Proves that the length of one lap is 1/4 mile, given the total distance and number of laps. -/
theorem lap_length (total_distance : ℚ) (num_laps : ℕ) :
  total_distance = 13/4 ∧ num_laps = 13 →
  total_distance / num_laps = 1/4 := by
sorry

end NUMINAMATH_CALUDE_lap_length_l1217_121728


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l1217_121756

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin x + 4 * Real.cos x
  ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l1217_121756


namespace NUMINAMATH_CALUDE_set_operations_l1217_121754

-- Define the sets A and B
def A : Set ℝ := {x | x - 1 ≥ 0}
def B : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2}) ∧
  ((Aᶜ ∩ Bᶜ) = {x : ℝ | x < -1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1217_121754


namespace NUMINAMATH_CALUDE_paint_cost_exceeds_budget_l1217_121784

/-- Represents the paint requirements for a mansion --/
structure MansionPaint where
  bedroom_count : Nat
  bathroom_count : Nat
  kitchen_count : Nat
  living_room_count : Nat
  dining_room_count : Nat
  study_room_count : Nat
  bedroom_paint : Nat
  bathroom_paint : Nat
  kitchen_paint : Nat
  living_room_paint : Nat
  dining_room_paint : Nat
  study_room_paint : Nat
  colored_paint_price : Nat
  white_paint_can_size : Nat
  white_paint_can_price : Nat
  budget : Nat

/-- Calculates the total cost of paint for the mansion --/
def total_paint_cost (m : MansionPaint) : Nat :=
  let colored_paint_gallons := 
    m.bedroom_count * m.bedroom_paint +
    m.kitchen_count * m.kitchen_paint +
    m.living_room_count * m.living_room_paint +
    m.dining_room_count * m.dining_room_paint +
    m.study_room_count * m.study_room_paint
  let white_paint_gallons := m.bathroom_count * m.bathroom_paint
  let white_paint_cans := (white_paint_gallons + m.white_paint_can_size - 1) / m.white_paint_can_size
  colored_paint_gallons * m.colored_paint_price + white_paint_cans * m.white_paint_can_price

/-- Theorem stating that the total paint cost exceeds the budget --/
theorem paint_cost_exceeds_budget (m : MansionPaint) 
  (h : m = { bedroom_count := 5, bathroom_count := 10, kitchen_count := 1, 
             living_room_count := 2, dining_room_count := 1, study_room_count := 1,
             bedroom_paint := 3, bathroom_paint := 2, kitchen_paint := 4,
             living_room_paint := 6, dining_room_paint := 4, study_room_paint := 3,
             colored_paint_price := 18, white_paint_can_size := 3, 
             white_paint_can_price := 40, budget := 500 }) : 
  total_paint_cost m > m.budget := by
  sorry


end NUMINAMATH_CALUDE_paint_cost_exceeds_budget_l1217_121784


namespace NUMINAMATH_CALUDE_binomial_coefficient_recurrence_l1217_121788

theorem binomial_coefficient_recurrence (n r : ℕ) (h1 : n > 0) (h2 : r > 0) (h3 : n > r) :
  Nat.choose n r = Nat.choose (n - 1) r + Nat.choose (n - 1) (r - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_recurrence_l1217_121788


namespace NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l1217_121737

/-- Given an n-th degree polynomial P(x) such that P(k) = 1 / C_n^k for k = 0, 1, 2, ..., n,
    prove that P(n+1) = 0 if n is odd and P(n+1) = 1 if n is even. -/
theorem polynomial_value_at_n_plus_one (n : ℕ) (P : ℝ → ℝ) :
  (∀ k : ℕ, k ≤ n → P k = 1 / (n.choose k)) →
  P (n + 1) = if n % 2 = 1 then 0 else 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_n_plus_one_l1217_121737


namespace NUMINAMATH_CALUDE_big_boxes_count_l1217_121733

theorem big_boxes_count (dolls_per_big_box : ℕ) (dolls_per_small_box : ℕ) 
  (small_box_count : ℕ) (total_dolls : ℕ) (h1 : dolls_per_big_box = 7) 
  (h2 : dolls_per_small_box = 4) (h3 : small_box_count = 9) (h4 : total_dolls = 71) :
  ∃ (big_box_count : ℕ), big_box_count * dolls_per_big_box + 
    small_box_count * dolls_per_small_box = total_dolls ∧ big_box_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_big_boxes_count_l1217_121733


namespace NUMINAMATH_CALUDE_remainder_problem_l1217_121739

theorem remainder_problem (x : ℕ+) : 
  203 % x.val = 13 ∧ 298 % x.val = 13 → x.val = 19 ∨ x.val = 95 :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l1217_121739


namespace NUMINAMATH_CALUDE_zoo_theorem_l1217_121709

def zoo_problem (tiger_enclosures : ℕ) (zebra_enclosures_per_tiger : ℕ) 
  (giraffe_enclosures_multiplier : ℕ) (tigers_per_enclosure : ℕ) 
  (zebras_per_enclosure : ℕ) (giraffes_per_enclosure : ℕ) : Prop :=
  let total_zebra_enclosures := tiger_enclosures * zebra_enclosures_per_tiger
  let total_giraffe_enclosures := total_zebra_enclosures * giraffe_enclosures_multiplier
  let total_tigers := tiger_enclosures * tigers_per_enclosure
  let total_zebras := total_zebra_enclosures * zebras_per_enclosure
  let total_giraffes := total_giraffe_enclosures * giraffes_per_enclosure
  let total_animals := total_tigers + total_zebras + total_giraffes
  total_animals = 144

theorem zoo_theorem : 
  zoo_problem 4 2 3 4 10 2 := by
  sorry

end NUMINAMATH_CALUDE_zoo_theorem_l1217_121709


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l1217_121711

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to its scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation (463.4 * 10^9) = ScientificNotation.mk 4.634 11 sorry :=
sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l1217_121711


namespace NUMINAMATH_CALUDE_only_first_proposition_correct_l1217_121776

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem only_first_proposition_correct 
  (m l : Line) (α β : Plane) 
  (h_diff_lines : m ≠ l) 
  (h_diff_planes : α ≠ β) :
  ((perpendicular_plane_line α l ∧ parallel_plane_line α m → perpendicular l m) ∧
   ¬(parallel m l ∧ line_in_plane m α → parallel_plane_line α l) ∧
   ¬(perpendicular_planes α β ∧ line_in_plane m α ∧ line_in_plane l β → perpendicular m l) ∧
   ¬(perpendicular m l ∧ line_in_plane m α ∧ line_in_plane l β → perpendicular_planes α β)) :=
by sorry

end NUMINAMATH_CALUDE_only_first_proposition_correct_l1217_121776


namespace NUMINAMATH_CALUDE_environmental_protection_contest_l1217_121795

theorem environmental_protection_contest (A B C : ℝ) 
  (hA : A = 3/4)
  (hAC : (1 - A) * (1 - C) = 1/12)
  (hBC : B * C = 1/4)
  (hIndep : ∀ X Y : ℝ, X * Y = X * Y) : 
  A * B * C + (1 - A) * B * C + A * (1 - B) * C + A * B * (1 - C) = 21/32 := by
  sorry

end NUMINAMATH_CALUDE_environmental_protection_contest_l1217_121795


namespace NUMINAMATH_CALUDE_min_square_area_is_121_l1217_121741

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The set of rectangles given in the problem -/
def problem_rectangles : List Rectangle := [
  { width := 2, height := 3 },
  { width := 3, height := 4 },
  { width := 1, height := 4 }
]

/-- 
  Given a list of rectangles, computes the smallest possible side length of a square 
  that can contain all rectangles without overlapping
-/
def min_square_side (rectangles : List Rectangle) : ℕ :=
  sorry

/-- 
  Theorem: The smallest possible area of a square containing the given rectangles 
  without overlapping is 121
-/
theorem min_square_area_is_121 : 
  (min_square_side problem_rectangles) ^ 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_min_square_area_is_121_l1217_121741


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1217_121718

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1217_121718


namespace NUMINAMATH_CALUDE_inequality_proof_l1217_121797

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  a / (b + c^2) + b / (c + a^2) + c / (a + b^2) ≥ 9/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1217_121797


namespace NUMINAMATH_CALUDE_ellipse_equal_angles_point_l1217_121759

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Defines the equation of the ellipse -/
def onEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Defines a chord passing through a given point -/
def isChord (e : Ellipse) (a b f : Point) : Prop :=
  onEllipse e a ∧ onEllipse e b ∧ ∃ t : ℝ, f = Point.mk (t * a.x + (1 - t) * b.x) (t * a.y + (1 - t) * b.y)

/-- Defines the property of equal angles -/
def equalAngles (p f a b : Point) : Prop :=
  (a.y - p.y) * (b.x - p.x) = (b.y - p.y) * (a.x - p.x)

/-- Main theorem statement -/
theorem ellipse_equal_angles_point :
  ∀ (e : Ellipse),
    e.a = 2 ∧ e.b = 1 →
    ∀ (f : Point),
      f.x = Real.sqrt 3 ∧ f.y = 0 →
      ∃! (p : Point),
        p.x > 0 ∧ p.y = 0 ∧
        (∀ (a b : Point), isChord e a b f → equalAngles p f a b) ∧
        p.x = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equal_angles_point_l1217_121759


namespace NUMINAMATH_CALUDE_cube_volume_doubled_edges_l1217_121716

/-- Given a cube, doubling each edge results in a volume 8 times larger than the original. -/
theorem cube_volume_doubled_edges (a : ℝ) (ha : a > 0) :
  (2 * a)^3 = 8 * a^3 := by sorry

end NUMINAMATH_CALUDE_cube_volume_doubled_edges_l1217_121716


namespace NUMINAMATH_CALUDE_parabola_vertex_l1217_121746

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := (x - 2)^2 - 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, -3)

/-- Theorem: The vertex of the parabola y = (x-2)^2 - 3 is (2, -3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1217_121746


namespace NUMINAMATH_CALUDE_expand_expression_l1217_121720

theorem expand_expression (x : ℝ) : (15 * x^2 + 5 - 3 * x) * 3 * x^3 = 45 * x^5 - 9 * x^4 + 15 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1217_121720


namespace NUMINAMATH_CALUDE_frieda_corner_probability_l1217_121796

/-- Represents the different types of squares on the 4x4 grid -/
inductive GridSquare
| Corner
| Edge
| Center

/-- Represents the possible directions of movement -/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents the state of Frieda on the grid -/
structure FriedaState :=
(position : GridSquare)
(hops : Nat)

/-- The probability of reaching a corner square within n hops -/
def probability_reach_corner (n : Nat) (start : GridSquare) : Rat :=
sorry

/-- The main theorem stating the probability of reaching a corner within 5 hops -/
theorem frieda_corner_probability :
  probability_reach_corner 5 GridSquare.Edge = 299 / 1024 :=
sorry

end NUMINAMATH_CALUDE_frieda_corner_probability_l1217_121796


namespace NUMINAMATH_CALUDE_sin_alpha_plus_beta_l1217_121742

theorem sin_alpha_plus_beta (α β : Real) 
  (h1 : Real.sin α + Real.cos β = 1) 
  (h2 : Real.cos α + Real.sin β = 0) : 
  Real.sin (α + β) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_beta_l1217_121742


namespace NUMINAMATH_CALUDE_hash_two_three_l1217_121730

-- Define the operation #
def hash (a b : ℕ) : ℕ := a * b - b + b^2

-- Theorem statement
theorem hash_two_three : hash 2 3 = 12 := by sorry

end NUMINAMATH_CALUDE_hash_two_three_l1217_121730


namespace NUMINAMATH_CALUDE_unique_congruence_l1217_121787

theorem unique_congruence (n : ℤ) : 
  12 ≤ n ∧ n ≤ 18 ∧ n ≡ 9001 [ZMOD 7] → n = 13 := by
sorry

end NUMINAMATH_CALUDE_unique_congruence_l1217_121787


namespace NUMINAMATH_CALUDE_triangle_tangent_inequality_l1217_121734

theorem triangle_tangent_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.tan A ^ 2 + Real.tan B ^ 2 + Real.tan C ^ 2 ≥ 
  Real.tan A * Real.tan B + Real.tan B * Real.tan C + Real.tan C * Real.tan A :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_inequality_l1217_121734


namespace NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l1217_121761

/-- Given complex numbers p, q, and r forming an equilateral triangle with side length 24,
    if |p + q + r| = 48, then |pq + pr + qr| = 768 -/
theorem equilateral_triangle_sum_product (p q r : ℂ) :
  (Complex.abs (p - q) = 24) →
  (Complex.abs (q - r) = 24) →
  (Complex.abs (r - p) = 24) →
  (Complex.abs (p + q + r) = 48) →
  Complex.abs (p*q + q*r + r*p) = 768 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l1217_121761
