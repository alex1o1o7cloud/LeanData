import Mathlib

namespace NUMINAMATH_CALUDE_total_informed_is_258_l4086_408614

/-- Represents the number of people in the initial group -/
def initial_group : ℕ := 6

/-- Represents the number of people each person calls -/
def calls_per_person : ℕ := 6

/-- Calculates the total number of people informed after two rounds of calls -/
def total_informed : ℕ := 
  initial_group + 
  (initial_group * calls_per_person) + 
  (initial_group * calls_per_person * calls_per_person)

/-- Theorem stating that the total number of people informed is 258 -/
theorem total_informed_is_258 : total_informed = 258 := by
  sorry

end NUMINAMATH_CALUDE_total_informed_is_258_l4086_408614


namespace NUMINAMATH_CALUDE_locus_of_P_l4086_408611

noncomputable def ellipse (x y : ℝ) : Prop := x^2/20 + y^2/16 = 1

def on_ellipse (M : ℝ × ℝ) : Prop := ellipse M.1 M.2

def intersect_x_axis (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 0 ∧ ellipse B.1 0 ∧ A.2 = 0 ∧ B.2 = 0

def tangent_line (l : ℝ → ℝ) (M : ℝ × ℝ) : Prop :=
  on_ellipse M ∧ ∀ x, l x = (M.2 / M.1) * (x - M.1) + M.2

def perpendicular_intersect (A B C D : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  C.1 = A.1 ∧ D.1 = B.1 ∧ l C.1 = C.2 ∧ l D.1 = D.2

def line_intersection (C B A D : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ, Q = (t * C.1 + (1 - t) * B.1, t * C.2 + (1 - t) * B.2) ∧
             Q = (s * A.1 + (1 - s) * D.1, s * A.2 + (1 - s) * D.2)

def symmetric_point (P Q M : ℝ × ℝ) : Prop :=
  P.1 + Q.1 = 2 * M.1 ∧ P.2 + Q.2 = 2 * M.2

theorem locus_of_P (A B M C D Q P : ℝ × ℝ) (l : ℝ → ℝ) :
  intersect_x_axis A B →
  on_ellipse M →
  M ≠ A →
  M ≠ B →
  tangent_line l M →
  perpendicular_intersect A B C D l →
  line_intersection C B A D Q →
  symmetric_point P Q M →
  (P.1^2 / 20 + P.2^2 / 36 = 1 ∧ P.2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_P_l4086_408611


namespace NUMINAMATH_CALUDE_pascal_triangle_34th_row_23rd_number_l4086_408652

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The 34th row of Pascal's triangle has 35 numbers -/
def row_length : ℕ := 35

/-- The row number (0-indexed) corresponding to a row with 35 numbers -/
def row_number : ℕ := row_length - 1

/-- The position (0-indexed) of the number we're looking for -/
def position : ℕ := 22

theorem pascal_triangle_34th_row_23rd_number :
  binomial row_number position = 64512240 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_34th_row_23rd_number_l4086_408652


namespace NUMINAMATH_CALUDE_tonys_laundry_problem_l4086_408685

/-- The problem of determining the weight of shirts in Tony's laundry. -/
theorem tonys_laundry_problem (
  wash_limit : ℕ)
  (sock_weight pants_weight shorts_weight underwear_weight : ℕ)
  (num_socks num_underwear : ℕ)
  (total_weight : ℕ → ℕ → ℕ → ℕ → ℕ) :
  wash_limit = 50 →
  sock_weight = 2 →
  pants_weight = 10 →
  shorts_weight = 8 →
  underwear_weight = 4 →
  num_socks = 3 →
  num_underwear = 4 →
  total_weight sock_weight pants_weight shorts_weight underwear_weight =
    sock_weight * num_socks + pants_weight + shorts_weight + underwear_weight * num_underwear →
  wash_limit - total_weight sock_weight pants_weight shorts_weight underwear_weight = 10 :=
by sorry

end NUMINAMATH_CALUDE_tonys_laundry_problem_l4086_408685


namespace NUMINAMATH_CALUDE_rita_swimming_months_l4086_408601

def swimming_months (total_required : ℕ) (completed : ℕ) (monthly_practice : ℕ) : ℕ :=
  (total_required - completed + monthly_practice - 1) / monthly_practice

theorem rita_swimming_months :
  swimming_months 2500 300 300 = 8 := by sorry

end NUMINAMATH_CALUDE_rita_swimming_months_l4086_408601


namespace NUMINAMATH_CALUDE_intersection_A_B_l4086_408600

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l4086_408600


namespace NUMINAMATH_CALUDE_cindy_crayons_count_l4086_408697

/-- The number of crayons Karen has -/
def karen_crayons : ℕ := 639

/-- The difference between Karen's and Cindy's crayons -/
def difference : ℕ := 135

/-- The number of crayons Cindy has -/
def cindy_crayons : ℕ := karen_crayons - difference

theorem cindy_crayons_count : cindy_crayons = 504 := by
  sorry

end NUMINAMATH_CALUDE_cindy_crayons_count_l4086_408697


namespace NUMINAMATH_CALUDE_red_box_position_l4086_408672

theorem red_box_position (n : ℕ) (initial_position : ℕ) (h1 : n = 45) (h2 : initial_position = 29) :
  n + 1 - initial_position = 17 := by
sorry

end NUMINAMATH_CALUDE_red_box_position_l4086_408672


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l4086_408612

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ - 15 = 0 ∧ x₂^2 - 2*x₂ - 15 = 0 ∧ x₁ = 5 ∧ x₂ = -3) ∧
  (∃ y₁ y₂ : ℝ, 2*y₁^2 + 3*y₁ = 1 ∧ 2*y₂^2 + 3*y₂ = 1 ∧ 
   y₁ = (-3 + Real.sqrt 17) / 4 ∧ y₂ = (-3 - Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l4086_408612


namespace NUMINAMATH_CALUDE_discount_equation_l4086_408628

theorem discount_equation (original_price final_price x : ℝ) 
  (h_original : original_price = 200)
  (h_final : final_price = 162)
  (h_positive : 0 < x ∧ x < 1) :
  original_price * (1 - x)^2 = final_price :=
sorry

end NUMINAMATH_CALUDE_discount_equation_l4086_408628


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l4086_408646

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.1 = 360) → total_land = 4000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l4086_408646


namespace NUMINAMATH_CALUDE_ice_cream_frozen_yoghurt_cost_difference_is_28_31_l4086_408677

/-- Calculates the difference between ice cream and frozen yoghurt costs --/
def ice_cream_frozen_yoghurt_cost_difference : ℝ :=
  let chocolate_ice_cream := 6 * 5 * (1 - 0.10)
  let vanilla_ice_cream := 4 * 4 * (1 - 0.07)
  let strawberry_frozen_yoghurt := 3 * 3 * (1 + 0.05)
  let mango_frozen_yoghurt := 2 * 2 * (1 + 0.03)
  let total_ice_cream := chocolate_ice_cream + vanilla_ice_cream
  let total_frozen_yoghurt := strawberry_frozen_yoghurt + mango_frozen_yoghurt
  total_ice_cream - total_frozen_yoghurt

/-- The difference between ice cream and frozen yoghurt costs is $28.31 --/
theorem ice_cream_frozen_yoghurt_cost_difference_is_28_31 :
  ice_cream_frozen_yoghurt_cost_difference = 28.31 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_frozen_yoghurt_cost_difference_is_28_31_l4086_408677


namespace NUMINAMATH_CALUDE_carly_lollipops_l4086_408644

/-- The number of grape lollipops -/
def grape_lollipops : ℕ := 7

/-- The total number of lollipops Carly has -/
def total_lollipops : ℕ := 42

/-- The number of non-cherry lollipop flavors -/
def non_cherry_flavors : ℕ := 3

theorem carly_lollipops :
  (total_lollipops / 2 = total_lollipops - total_lollipops / 2) ∧
  ((total_lollipops - total_lollipops / 2) / non_cherry_flavors = grape_lollipops) ∧
  (total_lollipops = 42) := by
  sorry

end NUMINAMATH_CALUDE_carly_lollipops_l4086_408644


namespace NUMINAMATH_CALUDE_valentines_given_proof_l4086_408631

/-- Represents the number of Valentines Mrs. Franklin initially had -/
def initial_valentines : ℕ := 58

/-- Represents the number of Valentines Mrs. Franklin has left -/
def remaining_valentines : ℕ := 16

/-- Represents the number of Valentines given to students -/
def valentines_given_to_students : ℕ := initial_valentines - remaining_valentines

/-- Theorem stating that the number of Valentines given to students
    is equal to the difference between initial and remaining Valentines -/
theorem valentines_given_proof :
  valentines_given_to_students = 42 :=
by sorry

end NUMINAMATH_CALUDE_valentines_given_proof_l4086_408631


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l4086_408668

/-- Proves that the interest rate is 6% given the conditions of the problem -/
theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (total_interest : ℝ) 
  (h1 : principal = 1000)
  (h2 : time = 8)
  (h3 : total_interest = 480)
  (h4 : total_interest = principal * (rate / 100) * time) :
  rate = 6 := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l4086_408668


namespace NUMINAMATH_CALUDE_smallest_a_value_l4086_408610

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem stating the smallest possible value of a for a parabola with given conditions -/
theorem smallest_a_value (p : Parabola) 
  (vertex_x : p.a * (3/5)^2 + p.b * (3/5) + p.c = -25/12)  -- vertex condition
  (pos_a : p.a > 0)  -- a > 0
  (int_sum : ∃ n : ℤ, p.a + p.b + p.c = n)  -- a + b + c is an integer
  : p.a ≥ 25/48 := by
  sorry


end NUMINAMATH_CALUDE_smallest_a_value_l4086_408610


namespace NUMINAMATH_CALUDE_prob_ratio_l4086_408660

/-- Represents the total number of cards in the box -/
def total_cards : ℕ := 50

/-- Represents the number of distinct card numbers -/
def distinct_numbers : ℕ := 10

/-- Represents the number of cards for each number -/
def cards_per_number : ℕ := 5

/-- Represents the number of cards drawn -/
def cards_drawn : ℕ := 5

/-- Calculates the probability of drawing five cards with the same number -/
def prob_same_number : ℚ :=
  (distinct_numbers : ℚ) / (total_cards.choose cards_drawn : ℚ)

/-- Calculates the probability of drawing four cards of one number and one card of a different number -/
def prob_four_and_one : ℚ :=
  ((distinct_numbers : ℚ) * (distinct_numbers - 1 : ℚ) * (cards_per_number : ℚ) * (cards_per_number : ℚ)) /
  (total_cards.choose cards_drawn : ℚ)

/-- Theorem stating the ratio of probabilities -/
theorem prob_ratio :
  prob_four_and_one / prob_same_number = 225 := by sorry

end NUMINAMATH_CALUDE_prob_ratio_l4086_408660


namespace NUMINAMATH_CALUDE_negative_three_is_monomial_l4086_408692

/-- A monomial is a constant term or a variable raised to a non-negative integer power -/
def IsMonomial (x : ℝ) : Prop :=
  x ≠ 0 ∨ ∃ (n : ℕ), x = 1 ∨ x = -1

/-- Prove that -3 is a monomial -/
theorem negative_three_is_monomial : IsMonomial (-3) := by
  sorry

end NUMINAMATH_CALUDE_negative_three_is_monomial_l4086_408692


namespace NUMINAMATH_CALUDE_mindy_emails_l4086_408603

theorem mindy_emails (phone_messages : ℕ) (emails : ℕ) : 
  emails = 9 * phone_messages - 7 →
  emails + phone_messages = 93 →
  emails = 83 := by
sorry

end NUMINAMATH_CALUDE_mindy_emails_l4086_408603


namespace NUMINAMATH_CALUDE_parabola_vertex_l4086_408624

/-- The vertex of the parabola y = 3x^2 - 6x + 2 is (1, -1) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 3 * x^2 - 6 * x + 2 → (1, -1) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l4086_408624


namespace NUMINAMATH_CALUDE_concert_processing_fee_percentage_l4086_408645

theorem concert_processing_fee_percentage
  (ticket_price : ℝ)
  (parking_fee : ℝ)
  (entrance_fee : ℝ)
  (total_cost : ℝ)
  (h1 : ticket_price = 50)
  (h2 : parking_fee = 10)
  (h3 : entrance_fee = 5)
  (h4 : total_cost = 135)
  : (total_cost - (2 * ticket_price + 2 * entrance_fee + parking_fee)) / (2 * ticket_price) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_concert_processing_fee_percentage_l4086_408645


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l4086_408627

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
  (5 : ℕ)^(x.val) - (3 : ℕ)^(y.val) = (z.val)^2 →
  x = 2 ∧ y = 2 ∧ z = 4 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l4086_408627


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l4086_408687

/-- The quadratic equation x^2 + (2k+1)x + k^2 + 1 = 0 has two distinct real roots -/
def has_distinct_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0 ∧ x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0

/-- The product of the roots of the quadratic equation is 5 -/
def roots_product_is_5 (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ * x₂ = 5 ∧ x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0 ∧ x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0

theorem quadratic_equation_properties :
  (∀ k : ℝ, has_distinct_roots k ↔ k > 3/4) ∧
  (∀ k : ℝ, roots_product_is_5 k → k = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l4086_408687


namespace NUMINAMATH_CALUDE_mongolia_1980_imo_host_l4086_408650

/-- Represents countries in East Asia -/
inductive EastAsianCountry
  | China
  | Japan
  | Mongolia
  | NorthKorea
  | SouthKorea
  | Taiwan

/-- Represents the International Mathematical Olympiad event -/
structure IMOEvent where
  year : Nat
  host : EastAsianCountry
  canceled : Bool

/-- The 1980 IMO event -/
def imo1980 : IMOEvent :=
  { year := 1980
  , host := EastAsianCountry.Mongolia
  , canceled := true }

/-- Theorem stating that Mongolia was the scheduled host of the canceled 1980 IMO -/
theorem mongolia_1980_imo_host :
  imo1980.year = 1980 ∧
  imo1980.host = EastAsianCountry.Mongolia ∧
  imo1980.canceled = true :=
by sorry

end NUMINAMATH_CALUDE_mongolia_1980_imo_host_l4086_408650


namespace NUMINAMATH_CALUDE_samantha_birth_year_proof_l4086_408637

/-- The year when the first AMC 8 was held -/
def first_amc8_year : ℕ := 1985

/-- The number of the AMC 8 in which Samantha participated -/
def samantha_amc8_number : ℕ := 7

/-- Samantha's age when she participated in the AMC 8 -/
def samantha_age_at_amc8 : ℕ := 12

/-- The year when Samantha was born -/
def samantha_birth_year : ℕ := 1979

theorem samantha_birth_year_proof :
  samantha_birth_year = first_amc8_year + (samantha_amc8_number - 1) - samantha_age_at_amc8 :=
by sorry

end NUMINAMATH_CALUDE_samantha_birth_year_proof_l4086_408637


namespace NUMINAMATH_CALUDE_multiples_of_four_l4086_408674

theorem multiples_of_four (n : ℕ) : 
  n ≤ 104 → 
  (∃ (k : ℕ), k = 24 ∧ 
    (∀ (i : ℕ), i ≤ k → 
      ∃ (m : ℕ), m * 4 = n + (i - 1) * 4 ∧ m * 4 ≤ 104)) → 
  n = 88 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_four_l4086_408674


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l4086_408602

theorem round_trip_average_speed (n : ℝ) : 
  let distance := n / 1000 -- distance in km
  let time_west := n / 30000 -- time for westward journey in hours
  let time_east := n / 3000 -- time for eastward journey in hours
  let time_wait := 0.5 -- waiting time in hours
  let total_distance := 2 * distance -- total round trip distance
  let total_time := time_west + time_east + time_wait -- total time for round trip
  total_distance / total_time = (60 * n) / (11 * n + 150000) := by
sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l4086_408602


namespace NUMINAMATH_CALUDE_theater_attendance_l4086_408605

theorem theater_attendance (adult_price child_price total_people total_revenue : ℕ)
  (h1 : adult_price = 8)
  (h2 : child_price = 1)
  (h3 : total_people = 22)
  (h4 : total_revenue = 50) :
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_revenue ∧
    children = 18 := by
  sorry

end NUMINAMATH_CALUDE_theater_attendance_l4086_408605


namespace NUMINAMATH_CALUDE_pqr_positive_iff_p_q_r_positive_l4086_408620

theorem pqr_positive_iff_p_q_r_positive
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (P : ℝ) (hP : P = a + b - c)
  (Q : ℝ) (hQ : Q = b + c - a)
  (R : ℝ) (hR : R = c + a - b) :
  P * Q * R > 0 ↔ P > 0 ∧ Q > 0 ∧ R > 0 := by
sorry

end NUMINAMATH_CALUDE_pqr_positive_iff_p_q_r_positive_l4086_408620


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4086_408661

/-- The function f(x) = a^(x-1) + 3 passes through the point (1, 4) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4086_408661


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_integers_l4086_408675

def is_all_ones (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1

theorem infinitely_many_divisible_integers :
  ∀ k : ℕ, ∃ n : ℕ, 
    n > k ∧ 
    is_all_ones n ∧ 
    n % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_integers_l4086_408675


namespace NUMINAMATH_CALUDE_robin_albums_l4086_408678

theorem robin_albums (total_pictures : ℕ) (pictures_per_album : ℕ) (h1 : total_pictures = 40) (h2 : pictures_per_album = 8) : 
  total_pictures / pictures_per_album = 5 := by
  sorry

end NUMINAMATH_CALUDE_robin_albums_l4086_408678


namespace NUMINAMATH_CALUDE_cube_face_perimeter_l4086_408673

/-- Given a cube with volume 1000 cm³, prove that the perimeter of one of its faces is 40 cm -/
theorem cube_face_perimeter (V : ℝ) (h : V = 1000) : 
  4 * (V ^ (1/3 : ℝ)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l4086_408673


namespace NUMINAMATH_CALUDE_translation_theorem_l4086_408682

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept - l.slope * dx + dy }

/-- The original line y = 2x - 3 -/
def original_line : Line :=
  { slope := 2,
    intercept := -3 }

/-- The amount of horizontal translation -/
def right_shift : ℝ := 2

/-- The amount of vertical translation -/
def up_shift : ℝ := 1

/-- The expected resulting line after translation -/
def expected_result : Line :=
  { slope := 2,
    intercept := -6 }

theorem translation_theorem :
  translate original_line right_shift up_shift = expected_result := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l4086_408682


namespace NUMINAMATH_CALUDE_function_properties_l4086_408695

noncomputable def f (a b x : ℝ) : ℝ := Real.log (x / 2) - a * x + b / x

theorem function_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x > 0, f a b x + f a b (4 / x) = 0) →
  (b = 4 * a ∧ (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0 ∧ f a b x₃ = 0) ↔ 0 < a ∧ a < 1/4) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4086_408695


namespace NUMINAMATH_CALUDE_even_sum_not_both_odd_l4086_408694

theorem even_sum_not_both_odd (n m : ℤ) :
  Even (n^2 + m^2 + n*m) → ¬(Odd n ∧ Odd m) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_not_both_odd_l4086_408694


namespace NUMINAMATH_CALUDE_system_solution_l4086_408625

theorem system_solution (a b x y z : ℝ) (ha : a ≠ 0) (hb : b ≠ 1) 
  (hyz : y ≠ z) (h2y3z : 2*y ≠ 3*z) (h3a2x2ay : 3*a^2*x ≠ 2*a*y) (hb_neq : b ≠ -19/15) :
  (a * x + z) / (y - z) = (1 + b) / (1 - b) ∧
  (2 * a * x - 3 * b) / (2 * y - 3 * z) = 1 ∧
  (5 * z - 4 * b) / (3 * a^2 * x - 2 * a * y) = b / a →
  x = 1/a ∧ y = 1 ∧ z = b := by
sorry

end NUMINAMATH_CALUDE_system_solution_l4086_408625


namespace NUMINAMATH_CALUDE_rita_breaststroke_hours_l4086_408671

/-- Calculates the hours of breaststroke completed by Rita --/
def breaststroke_hours (total_required : ℕ) (backstroke : ℕ) (butterfly : ℕ) (freestyle_sidestroke_per_month : ℕ) (months : ℕ) : ℕ :=
  total_required - (backstroke + butterfly + freestyle_sidestroke_per_month * months)

/-- Theorem stating that Rita completed 9 hours of breaststroke --/
theorem rita_breaststroke_hours : 
  breaststroke_hours 1500 50 121 220 6 = 9 := by
  sorry

#eval breaststroke_hours 1500 50 121 220 6

end NUMINAMATH_CALUDE_rita_breaststroke_hours_l4086_408671


namespace NUMINAMATH_CALUDE_total_potatoes_l4086_408662

def nancys_potatoes : ℕ := 6
def sandys_potatoes : ℕ := 7

theorem total_potatoes : nancys_potatoes + sandys_potatoes = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_potatoes_l4086_408662


namespace NUMINAMATH_CALUDE_negative_cube_squared_l4086_408676

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l4086_408676


namespace NUMINAMATH_CALUDE_intersection_of_p_and_q_when_a_is_one_range_of_a_for_not_p_sufficient_for_not_q_l4086_408619

/-- Proposition p: x^2 - 5ax + 4a^2 < 0, where a > 0 -/
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0 ∧ a > 0

/-- Proposition q: x^2 - 2x - 8 ≤ 0 and x^2 + 3x - 10 > 0 -/
def q (x : ℝ) : Prop := x^2 - 2*x - 8 ≤ 0 ∧ x^2 + 3*x - 10 > 0

/-- The solution set of p -/
def solution_set_p (a : ℝ) : Set ℝ := {x | p x a}

/-- The solution set of q -/
def solution_set_q : Set ℝ := {x | q x}

theorem intersection_of_p_and_q_when_a_is_one :
  (solution_set_p 1) ∩ solution_set_q = Set.Ioo 2 4 := by sorry

theorem range_of_a_for_not_p_sufficient_for_not_q :
  {a : ℝ | ∀ x, ¬(p x a) → ¬(q x)} = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_p_and_q_when_a_is_one_range_of_a_for_not_p_sufficient_for_not_q_l4086_408619


namespace NUMINAMATH_CALUDE_num_odd_factors_252_is_6_l4086_408681

/-- The number of odd factors of 252 -/
def num_odd_factors_252 : ℕ := sorry

/-- 252 is the product of its prime factors -/
axiom factorization_252 : 252 = 2^2 * 3^2 * 7

/-- Theorem stating that the number of odd factors of 252 is 6 -/
theorem num_odd_factors_252_is_6 : num_odd_factors_252 = 6 := by sorry

end NUMINAMATH_CALUDE_num_odd_factors_252_is_6_l4086_408681


namespace NUMINAMATH_CALUDE_proposition_a_necessary_not_sufficient_for_b_l4086_408635

theorem proposition_a_necessary_not_sufficient_for_b (h : ℝ) (h_pos : h > 0) :
  (∀ a b : ℝ, (|a - 1| < h ∧ |b - 1| < h) → |a - b| < 2 * h) ∧
  (∃ a b : ℝ, |a - b| < 2 * h ∧ ¬(|a - 1| < h ∧ |b - 1| < h)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_a_necessary_not_sufficient_for_b_l4086_408635


namespace NUMINAMATH_CALUDE_floor_abs_calculation_l4086_408653

theorem floor_abs_calculation : (((⌊|(-7.6 : ℝ)|⌋ : ℤ) + |⌊(-7.6 : ℝ)⌋|) : ℤ) * 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_calculation_l4086_408653


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l4086_408643

/-- The speed of a man in still water, given his downstream and upstream swimming times and distances. -/
theorem mans_speed_in_still_water 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (h1 : downstream_distance = 48) 
  (h2 : upstream_distance = 18) 
  (h3 : downstream_time = 3) 
  (h4 : upstream_time = 3) : 
  ∃ (speed : ℝ), speed = 11 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l4086_408643


namespace NUMINAMATH_CALUDE_jarek_calculation_l4086_408666

theorem jarek_calculation (x : ℝ) (h : 3 * x = 90) : x / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jarek_calculation_l4086_408666


namespace NUMINAMATH_CALUDE_scientific_notation_of_600_billion_l4086_408647

theorem scientific_notation_of_600_billion :
  let billion : ℕ := 10^9
  600 * billion = 6 * 10^11 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_600_billion_l4086_408647


namespace NUMINAMATH_CALUDE_sum_of_C_and_D_l4086_408663

/-- Represents a 4x4 table with numbers 1 to 4 -/
def Table := Fin 4 → Fin 4 → Fin 4

/-- Checks if a row contains all numbers from 1 to 4 -/
def validRow (t : Table) (row : Fin 4) : Prop :=
  ∀ n : Fin 4, ∃ col : Fin 4, t row col = n

/-- Checks if a column contains all numbers from 1 to 4 -/
def validColumn (t : Table) (col : Fin 4) : Prop :=
  ∀ n : Fin 4, ∃ row : Fin 4, t row col = n

/-- Checks if the table satisfies all given constraints -/
def validTable (t : Table) : Prop :=
  (∀ row : Fin 4, validRow t row) ∧
  (∀ col : Fin 4, validColumn t col) ∧
  t 0 0 = 1 ∧
  t 1 1 = 2 ∧
  t 3 3 = 4

theorem sum_of_C_and_D (t : Table) (h : validTable t) :
  t 1 2 + t 2 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_C_and_D_l4086_408663


namespace NUMINAMATH_CALUDE_tunnel_digging_problem_l4086_408655

theorem tunnel_digging_problem (total_length : ℝ) (team_a_rate : ℝ) (team_b_rate : ℝ) (remaining_distance : ℝ) :
  total_length = 1200 ∧ 
  team_a_rate = 12 ∧ 
  team_b_rate = 8 ∧ 
  remaining_distance = 200 →
  (total_length - remaining_distance) / (team_a_rate + team_b_rate) = 50 := by
sorry

end NUMINAMATH_CALUDE_tunnel_digging_problem_l4086_408655


namespace NUMINAMATH_CALUDE_sunflower_majority_day_two_l4086_408648

/-- Represents the proportion of sunflower seeds in the feeder on a given day -/
def sunflower_proportion (day : ℕ) : ℝ :=
  1 - (0.6 : ℝ) ^ day

/-- The daily seed mixture contains 40% sunflower seeds -/
axiom seed_mixture : (0.4 : ℝ) = 1 - (0.6 : ℝ)

/-- Birds eat 40% of sunflower seeds daily -/
axiom bird_consumption : (0.6 : ℝ) = 1 - (0.4 : ℝ)

/-- Theorem: On day 2, more than half the seeds are sunflower seeds -/
theorem sunflower_majority_day_two :
  sunflower_proportion 2 > (0.5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sunflower_majority_day_two_l4086_408648


namespace NUMINAMATH_CALUDE_hotpot_revenue_problem_l4086_408689

/-- Represents the revenue from different sources in a hotpot restaurant -/
structure HotpotRevenue where
  diningIn : ℝ
  takeout : ℝ
  stall : ℝ

/-- The revenue increase from different sources in July -/
structure JulyIncrease where
  diningIn : ℝ
  takeout : ℝ
  stall : ℝ

/-- Theorem representing the hotpot restaurant revenue problem -/
theorem hotpot_revenue_problem 
  (june : HotpotRevenue) 
  (july_increase : JulyIncrease) 
  (july : HotpotRevenue) :
  -- June revenue ratio condition
  june.diningIn / june.takeout = 3 / 5 ∧ 
  june.takeout / june.stall = 5 / 2 ∧
  -- July stall revenue increase condition
  july_increase.stall = 2 / 5 * (july_increase.diningIn + july_increase.takeout + july_increase.stall) ∧
  -- July stall revenue proportion condition
  july.stall / (july.diningIn + july.takeout + july.stall) = 7 / 20 ∧
  -- July dining in to takeout ratio condition
  july.diningIn / july.takeout = 8 / 5 ∧
  -- July revenue calculation
  july.diningIn = june.diningIn + july_increase.diningIn ∧
  july.takeout = june.takeout + july_increase.takeout ∧
  july.stall = june.stall + july_increase.stall
  →
  -- Conclusion: Additional takeout revenue in July compared to total July revenue
  july_increase.takeout / (july.diningIn + july.takeout + july.stall) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_hotpot_revenue_problem_l4086_408689


namespace NUMINAMATH_CALUDE_integral_inequality_l4086_408638

variables {a b : ℝ} (f : ℝ → ℝ)

/-- The main theorem statement -/
theorem integral_inequality
  (hab : 0 < a ∧ a < b)
  (hf : Continuous f)
  (hf_int : ∫ x in a..b, f x = 0) :
  ∫ x in a..b, ∫ y in a..b, f x * f y * Real.log (x + y) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_integral_inequality_l4086_408638


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l4086_408691

theorem quadratic_inequality_condition (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2)*x - 2*k + 4 < 0) ↔ -6 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l4086_408691


namespace NUMINAMATH_CALUDE_total_triangles_in_pattern_l4086_408607

/-- Represents the hexagonal pattern with triangular subdivisions -/
structure HexagonalPattern :=
  (small_triangles : ℕ)
  (medium_triangles : ℕ)
  (large_triangles : ℕ)
  (extra_large_triangles : ℕ)

/-- The total number of triangles in the hexagonal pattern -/
def total_triangles (pattern : HexagonalPattern) : ℕ :=
  pattern.small_triangles + pattern.medium_triangles + pattern.large_triangles + pattern.extra_large_triangles

/-- The specific hexagonal pattern described in the problem -/
def given_pattern : HexagonalPattern :=
  { small_triangles := 10,
    medium_triangles := 6,
    large_triangles := 3,
    extra_large_triangles := 1 }

theorem total_triangles_in_pattern :
  total_triangles given_pattern = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_in_pattern_l4086_408607


namespace NUMINAMATH_CALUDE_division_equality_l4086_408613

theorem division_equality : (999 - 99 + 9) / 9 = 101 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l4086_408613


namespace NUMINAMATH_CALUDE_binomial_expectation_variance_l4086_408656

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  X : ℝ → ℝ
  prob_zero : ℝ
  is_binomial : Prop
  prob_zero_eq : prob_zero = 1/3

/-- The expectation of a random variable -/
noncomputable def expectation (X : ℝ → ℝ) : ℝ := sorry

/-- The variance of a random variable -/
noncomputable def variance (X : ℝ → ℝ) : ℝ := sorry

theorem binomial_expectation_variance 
  (rv : BinomialRV) : 
  expectation (fun x => 3 * rv.X x + 2) = 4 ∧ 
  variance (fun x => 3 * rv.X x + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_variance_l4086_408656


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4086_408623

theorem quadratic_equation_roots (a : ℝ) : 
  (3 : ℝ)^2 + a * 3 - 2 * a = 0 → 
  ∃ b : ℝ, b^2 + a * b - 2 * a = 0 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4086_408623


namespace NUMINAMATH_CALUDE_set_operations_l4086_408641

def A : Set ℝ := {x | x < 0 ∨ x ≥ 2}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem set_operations :
  (A ∪ B = {x | x ≥ 2 ∨ x < 1}) ∧
  ((Aᶜ ∩ B) = {x | 0 ≤ x ∧ x < 1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l4086_408641


namespace NUMINAMATH_CALUDE_vectors_perpendicular_l4086_408642

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 1)

theorem vectors_perpendicular : a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_l4086_408642


namespace NUMINAMATH_CALUDE_work_multiple_l4086_408690

/-- Given that P people can complete a job in 8 days, 
    this theorem proves that 2P people can complete half the job in 2 days -/
theorem work_multiple (P : ℕ) : 
  (P * 8 : ℚ)⁻¹ * 2 * P * 2 = 1/2 := by sorry

end NUMINAMATH_CALUDE_work_multiple_l4086_408690


namespace NUMINAMATH_CALUDE_velociraptor_catch_up_time_l4086_408639

/-- The time it takes for a velociraptor to catch up to a person, given their respective speeds and initial head start. -/
theorem velociraptor_catch_up_time 
  (your_speed : ℝ)
  (velociraptor_speed : ℝ)
  (head_start_time : ℝ)
  (h1 : your_speed = 10)
  (h2 : velociraptor_speed = 15 * Real.sqrt 2)
  (h3 : head_start_time = 3) :
  (head_start_time * your_speed) / (velociraptor_speed / Real.sqrt 2 - your_speed) = 6 := by
  sorry

#check velociraptor_catch_up_time

end NUMINAMATH_CALUDE_velociraptor_catch_up_time_l4086_408639


namespace NUMINAMATH_CALUDE_min_value_of_expression_l4086_408651

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + y' = 1 → 1/(2*x') + 1/y' ≥ 1/(2*x) + 1/y) →
  1/(2*x) + 1/y = 3/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l4086_408651


namespace NUMINAMATH_CALUDE_ellipse_midpoint_y_coordinate_l4086_408629

/-- The y-coordinate of the midpoint M of PF, where F is a focus of the ellipse
    x^2/12 + y^2/3 = 1 and P is a point on the ellipse such that M lies on the y-axis. -/
theorem ellipse_midpoint_y_coordinate (x_P y_P : ℝ) (x_F y_F : ℝ) :
  x_P^2 / 12 + y_P^2 / 3 = 1 →  -- P is on the ellipse
  x_F^2 = 9 ∧ y_F = 0 →  -- F is a focus
  (x_P + x_F) / 2 = 0 →  -- M is on the y-axis
  ∃ (y_M : ℝ), y_M = (y_P + y_F) / 2 ∧ y_M^2 = 3 / 16 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_midpoint_y_coordinate_l4086_408629


namespace NUMINAMATH_CALUDE_max_sum_of_cubes_l4086_408636

theorem max_sum_of_cubes (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) : 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_cubes_l4086_408636


namespace NUMINAMATH_CALUDE_paint_for_smaller_statues_l4086_408659

-- Define the height of the original statue
def original_height : ℝ := 6

-- Define the height of the smaller statues
def small_height : ℝ := 2

-- Define the number of smaller statues
def num_statues : ℕ := 1080

-- Define the amount of paint needed for the original statue
def paint_for_original : ℝ := 1

-- Theorem statement
theorem paint_for_smaller_statues :
  (paint_for_original * (small_height / original_height)^2 * num_statues : ℝ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_paint_for_smaller_statues_l4086_408659


namespace NUMINAMATH_CALUDE_average_weight_abc_l4086_408606

theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 48 →
  (b + c) / 2 = 42 →
  b = 51 →
  (a + b + c) / 3 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_abc_l4086_408606


namespace NUMINAMATH_CALUDE_smallest_valid_n_l4086_408670

/-- Represents the graph structure with 8 vertices --/
def Graph := Fin 8 → Fin 8 → Bool

/-- The specific graph structure given in the problem --/
def problemGraph : Graph := sorry

/-- Checks if two numbers are coprime --/
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Checks if two numbers have a common divisor greater than 1 --/
def hasCommonDivisorGreaterThanOne (a b : ℕ) : Prop := ∃ (d : ℕ), d > 1 ∧ d ∣ a ∧ d ∣ b

/-- Represents a valid arrangement of numbers in the graph --/
def ValidArrangement (n : ℕ) (arr : Fin 8 → ℕ) : Prop :=
  (∀ i j, i ≠ j → arr i ≠ arr j) ∧
  (∀ i j, ¬problemGraph i j → coprime (arr i + arr j) n) ∧
  (∀ i j, problemGraph i j → hasCommonDivisorGreaterThanOne (arr i + arr j) n)

/-- The main theorem stating that 35 is the smallest valid n --/
theorem smallest_valid_n :
  (∃ (arr : Fin 8 → ℕ), ValidArrangement 35 arr) ∧
  (∀ n < 35, ¬∃ (arr : Fin 8 → ℕ), ValidArrangement n arr) := by sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l4086_408670


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4086_408618

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3 * y = 12 :=
by
  -- The unique solution is y = 2
  use 2
  constructor
  · -- Prove that y = 2 satisfies the equation
    simp
    norm_num
  · -- Prove uniqueness
    intro z hz
    -- Proof goes here
    sorry

#check absolute_value_equation_solution

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4086_408618


namespace NUMINAMATH_CALUDE_debby_tickets_spent_l4086_408693

theorem debby_tickets_spent (hat_tickets stuffed_animal_tickets yoyo_tickets : ℕ) 
  (h1 : hat_tickets = 2)
  (h2 : stuffed_animal_tickets = 10)
  (h3 : yoyo_tickets = 2) :
  hat_tickets + stuffed_animal_tickets + yoyo_tickets = 14 :=
by sorry

end NUMINAMATH_CALUDE_debby_tickets_spent_l4086_408693


namespace NUMINAMATH_CALUDE_small_slice_price_l4086_408617

/-- The price of a small slice of pizza given the following conditions:
  1. Large slices are sold for Rs. 250 each
  2. 5000 slices were sold in total
  3. Total revenue was Rs. 1,050,000
  4. 2000 small slices were sold
-/
theorem small_slice_price (large_slice_price : ℕ) (total_slices : ℕ) (total_revenue : ℕ) (small_slices : ℕ) :
  large_slice_price = 250 →
  total_slices = 5000 →
  total_revenue = 1050000 →
  small_slices = 2000 →
  ∃ (small_slice_price : ℕ),
    small_slice_price * small_slices + large_slice_price * (total_slices - small_slices) = total_revenue ∧
    small_slice_price = 150 :=
by sorry

end NUMINAMATH_CALUDE_small_slice_price_l4086_408617


namespace NUMINAMATH_CALUDE_log_one_over_81_base_3_l4086_408621

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_over_81_base_3 : log 3 (1/81) = -4 := by
  sorry

end NUMINAMATH_CALUDE_log_one_over_81_base_3_l4086_408621


namespace NUMINAMATH_CALUDE_type_a_lowest_price_lowest_price_value_l4086_408633

/-- Represents the types of pet food --/
inductive PetFoodType
  | A
  | B
  | C

/-- Calculates the final price of pet food after discounts, conversion, and tax --/
def finalPrice (type : PetFoodType) : ℝ :=
  let msrp : ℝ := match type with
    | PetFoodType.A => 45
    | PetFoodType.B => 55
    | PetFoodType.C => 50
  let regularDiscount : ℝ := match type with
    | PetFoodType.A => 0.15
    | PetFoodType.B => 0.25
    | PetFoodType.C => 0.30
  let additionalDiscount : ℝ := match type with
    | PetFoodType.A => 0.20
    | PetFoodType.B => 0.15
    | PetFoodType.C => 0.10
  let salesTax : ℝ := 0.07
  let exchangeRate : ℝ := 1.1
  
  msrp * (1 - regularDiscount) * (1 - additionalDiscount) * exchangeRate * (1 + salesTax)

/-- Theorem: Type A pet food has the lowest final price --/
theorem type_a_lowest_price :
  ∀ (type : PetFoodType), finalPrice PetFoodType.A ≤ finalPrice type :=
by sorry

/-- Corollary: The lowest final price is $36.02 --/
theorem lowest_price_value :
  finalPrice PetFoodType.A = 36.02 :=
by sorry

end NUMINAMATH_CALUDE_type_a_lowest_price_lowest_price_value_l4086_408633


namespace NUMINAMATH_CALUDE_count_valid_B_l4086_408615

def is_divisible_by_33 (n : ℕ) : Prop := n % 33 = 0

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def number_3A3B3 (A B : ℕ) : ℕ := 30303 + 1000 * A + 10 * B

theorem count_valid_B :
  ∃ (S : Finset ℕ),
    (∀ B ∈ S, digit B) ∧
    (∀ A, digit A → (is_divisible_by_33 (number_3A3B3 A B) ↔ B ∈ S)) ∧
    Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_count_valid_B_l4086_408615


namespace NUMINAMATH_CALUDE_log_equation_solution_l4086_408630

theorem log_equation_solution (x : ℝ) :
  x > 0 → (Real.log 4 / Real.log x = Real.log 3 / Real.log 27) → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4086_408630


namespace NUMINAMATH_CALUDE_triangle_max_area_l4086_408683

theorem triangle_max_area (a b : ℝ) (h1 : a + b = 4) (h2 : 0 < a) (h3 : 0 < b) : 
  (1/2 : ℝ) * a * b * Real.sin (π/6) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l4086_408683


namespace NUMINAMATH_CALUDE_probability_sum_less_2_or_greater_3_l4086_408667

/-- Represents a bag of balls with marks -/
structure Bag :=
  (total : ℕ)
  (zeros : ℕ)
  (ones : ℕ)
  (h_total : total = zeros + ones)

/-- Represents the number of balls drawn from the bag -/
def drawn : ℕ := 5

/-- The specific bag described in the problem -/
def problem_bag : Bag :=
  { total := 10
  , zeros := 5
  , ones := 5
  , h_total := rfl }

/-- The probability of drawing 5 balls with sum of marks less than 2 or greater than 3 -/
def probability (b : Bag) : ℚ :=
  38 / 63

theorem probability_sum_less_2_or_greater_3 :
  probability problem_bag = 38 / 63 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_less_2_or_greater_3_l4086_408667


namespace NUMINAMATH_CALUDE_average_senior_visitors_l4086_408665

/-- Represents the categories of visitors -/
inductive VisitorCategory
  | Adult
  | Student
  | Senior

/-- Represents the types of days -/
inductive DayType
  | Sunday
  | Other

/-- Average number of visitors for each day type -/
def averageVisitors (d : DayType) : ℕ :=
  match d with
  | DayType.Sunday => 150
  | DayType.Other => 120

/-- Ratio of visitors for each category on each day type -/
def visitorRatio (c : VisitorCategory) (d : DayType) : ℕ :=
  match d with
  | DayType.Sunday =>
    match c with
    | VisitorCategory.Adult => 5
    | VisitorCategory.Student => 3
    | VisitorCategory.Senior => 2
  | DayType.Other =>
    match c with
    | VisitorCategory.Adult => 4
    | VisitorCategory.Student => 3
    | VisitorCategory.Senior => 3

def daysInMonth : ℕ := 30
def sundaysInMonth : ℕ := 5
def otherDaysInMonth : ℕ := daysInMonth - sundaysInMonth

theorem average_senior_visitors :
  (sundaysInMonth * averageVisitors DayType.Sunday * visitorRatio VisitorCategory.Senior DayType.Sunday +
   otherDaysInMonth * averageVisitors DayType.Other * visitorRatio VisitorCategory.Senior DayType.Other) /
  daysInMonth = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_senior_visitors_l4086_408665


namespace NUMINAMATH_CALUDE_equation_solution_l4086_408649

theorem equation_solution (k : ℝ) : 
  (∀ x : ℝ, 5 * x + 3 * k = 24 ↔ 5 * x + 3 = 0) → k = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4086_408649


namespace NUMINAMATH_CALUDE_students_in_two_courses_l4086_408696

theorem students_in_two_courses 
  (total : ℕ) 
  (math : ℕ) 
  (chinese : ℕ) 
  (international : ℕ) 
  (all_three : ℕ) 
  (none : ℕ) 
  (h1 : total = 400) 
  (h2 : math = 169) 
  (h3 : chinese = 158) 
  (h4 : international = 145) 
  (h5 : all_three = 30) 
  (h6 : none = 20) : 
  ∃ (two_courses : ℕ), 
    two_courses = 32 ∧ 
    total = math + chinese + international - two_courses - 2 * all_three + none :=
by sorry

end NUMINAMATH_CALUDE_students_in_two_courses_l4086_408696


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l4086_408657

theorem shortest_side_of_right_triangle (a b c : ℝ) : 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → a ≤ b ∧ a ≤ c := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l4086_408657


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l4086_408622

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 1 ∧ b = -4 ∧ c = -12 → abs (r₁ - r₂) = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l4086_408622


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l4086_408688

/-- A geometric sequence with common ratio r -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_value (a : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  a 4 + a 8 = -2 →
  a 6 * (a 2 + 2 * a 6 + a 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_value_l4086_408688


namespace NUMINAMATH_CALUDE_gcd_problem_l4086_408679

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 714 * k) :
  Int.gcd (5 * b^3 + 2 * b^2 + 6 * b + 102) b = 102 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l4086_408679


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_l4086_408669

/-- Represents a triangle with integer side lengths -/
structure IntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- A triangle with one side four times another and the third side 20 -/
def SpecialTriangle (x : ℕ) : IntegerTriangle where
  a := x
  b := 4 * x
  c := 20
  triangle_inequality := sorry

/-- The perimeter of a triangle -/
def perimeter (t : IntegerTriangle) : ℕ := t.a + t.b + t.c

/-- Theorem stating the maximum perimeter of the special triangle -/
theorem max_perimeter_special_triangle :
  ∃ (t : IntegerTriangle), (∃ x, t = SpecialTriangle x) ∧
    (∀ (t' : IntegerTriangle), (∃ x, t' = SpecialTriangle x) → perimeter t' ≤ perimeter t) ∧
    perimeter t = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_l4086_408669


namespace NUMINAMATH_CALUDE_auction_bid_relationship_l4086_408634

/-- Joe's bid at the auction -/
def joes_bid : ℝ := 160000

/-- Nelly's winning bid at the auction -/
def nellys_bid : ℝ := 482000

/-- Theorem stating the relationship between Joe's and Nelly's bids -/
theorem auction_bid_relationship : 
  nellys_bid = 3 * joes_bid + 2000 ∧ joes_bid = 160000 := by
  sorry

end NUMINAMATH_CALUDE_auction_bid_relationship_l4086_408634


namespace NUMINAMATH_CALUDE_age_problem_l4086_408658

/-- Given the ages of Matt, Kaylee, Bella, and Alex, prove that they satisfy the given conditions -/
theorem age_problem (matt_age kaylee_age bella_age alex_age : ℕ) : 
  matt_age = 5 ∧ 
  kaylee_age + 7 = 3 * matt_age ∧ 
  kaylee_age + bella_age = matt_age + 9 ∧ 
  bella_age = alex_age + 3 →
  kaylee_age = 8 ∧ matt_age = 5 ∧ bella_age = 6 ∧ alex_age = 3 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l4086_408658


namespace NUMINAMATH_CALUDE_austen_gathering_handshakes_l4086_408654

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : ℕ
  men_shake_all_but_spouse : Bool
  women_shake_count : ℕ

/-- Calculates the total number of handshakes in the gathering -/
def total_handshakes (g : Gathering) : ℕ :=
  let total_people := 2 * g.couples
  let men_count := g.couples
  let women_count := g.couples
  let men_handshakes := men_count.choose 2
  let men_women_handshakes := if g.men_shake_all_but_spouse then men_count * (women_count - 1) else 0
  men_handshakes + men_women_handshakes + g.women_shake_count

theorem austen_gathering_handshakes :
  let g : Gathering := { couples := 15, men_shake_all_but_spouse := true, women_shake_count := 1 }
  total_handshakes g = 316 := by
  sorry

end NUMINAMATH_CALUDE_austen_gathering_handshakes_l4086_408654


namespace NUMINAMATH_CALUDE_store_coupon_distribution_l4086_408616

/-- Calculates the number of coupons per remaining coloring book -/
def coupons_per_book (initial_stock : ℚ) (books_sold : ℚ) (total_coupons : ℕ) : ℚ :=
  total_coupons / (initial_stock - books_sold)

/-- Proves that given the problem conditions, the number of coupons per remaining book is 4 -/
theorem store_coupon_distribution :
  coupons_per_book 40 20 80 = 4 := by
  sorry

#eval coupons_per_book 40 20 80

end NUMINAMATH_CALUDE_store_coupon_distribution_l4086_408616


namespace NUMINAMATH_CALUDE_polynomial_pell_equation_l4086_408604

theorem polynomial_pell_equation (a b : ℤ) :
  (∃ (p q : ℝ → ℝ), ∀ x : ℝ, 
    (∃ (cp cq : ℤ → ℝ), (∀ n : ℤ, p x = cp n * x^n) ∧ (∀ n : ℤ, q x = cq n * x^n)) ∧ 
    q ≠ 0 ∧ 
    (p x)^2 - (x^2 + a*x + b) * (q x)^2 = 1) ↔ 
  (a % 2 = 1 ∧ b = (a^2 - 1) / 4) ∨ 
  (a % 2 = 0 ∧ ∃ k : ℤ, b = a^2 / 4 + k ∧ 2 % k = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_pell_equation_l4086_408604


namespace NUMINAMATH_CALUDE_min_value_a5_plus_a6_l4086_408608

/-- A positive arithmetic geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem min_value_a5_plus_a6 (a : ℕ → ℝ) 
    (h_seq : ArithmeticGeometricSequence a) 
    (h_cond : a 4 + a 3 - a 2 - a 1 = 1) :
    ∃ (min_val : ℝ), min_val = 4 ∧ 
    (∀ a5 a6, a 5 = a5 → a 6 = a6 → a5 + a6 ≥ min_val) ∧
    (∃ a5 a6, a 5 = a5 ∧ a 6 = a6 ∧ a5 + a6 = min_val) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a5_plus_a6_l4086_408608


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l4086_408680

open Real

theorem trigonometric_equation_solution (x : ℝ) : 
  (abs (cos x) - cos (3 * x)) / (cos x * sin (2 * x)) = 2 / sqrt 3 ↔ 
  (∃ k : ℤ, x = π / 6 + 2 * k * π) ∨ 
  (∃ k : ℤ, x = 5 * π / 6 + 2 * k * π) ∨ 
  (∃ k : ℤ, x = 4 * π / 3 + 2 * k * π) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l4086_408680


namespace NUMINAMATH_CALUDE_empty_set_proof_l4086_408699

theorem empty_set_proof : {x : ℝ | x^2 - x + 1 = 0} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_set_proof_l4086_408699


namespace NUMINAMATH_CALUDE_jacoby_needs_3214_l4086_408640

/-- The amount Jacoby needs for his trip to Brickville -/
def tripCost : ℕ := 5000

/-- Jacoby's hourly wage -/
def hourlyWage : ℕ := 20

/-- Hours Jacoby worked -/
def hoursWorked : ℕ := 10

/-- Price of each cookie -/
def cookiePrice : ℕ := 4

/-- Number of cookies sold -/
def cookiesSold : ℕ := 24

/-- Cost of lottery ticket -/
def lotteryCost : ℕ := 10

/-- Lottery winnings -/
def lotteryWin : ℕ := 500

/-- Gift amount from each sister -/
def sisterGift : ℕ := 500

/-- Number of sisters who gave gifts -/
def numSisters : ℕ := 2

/-- Calculate the remaining amount Jacoby needs for his trip -/
def remainingAmount : ℕ :=
  tripCost - (
    hourlyWage * hoursWorked +
    cookiePrice * cookiesSold +
    lotteryWin +
    sisterGift * numSisters -
    lotteryCost
  )

theorem jacoby_needs_3214 : remainingAmount = 3214 := by
  sorry

end NUMINAMATH_CALUDE_jacoby_needs_3214_l4086_408640


namespace NUMINAMATH_CALUDE_correct_number_of_seasons_l4086_408684

/-- Represents the number of seasons in a TV show. -/
def num_seasons : ℕ := 5

/-- Cost per episode for the first season. -/
def first_season_cost : ℕ := 100000

/-- Cost per episode for other seasons. -/
def other_season_cost : ℕ := 200000

/-- Number of episodes in the first season. -/
def first_season_episodes : ℕ := 12

/-- Number of episodes in middle seasons. -/
def middle_season_episodes : ℕ := 18

/-- Number of episodes in the last season. -/
def last_season_episodes : ℕ := 24

/-- Total cost of producing all episodes. -/
def total_cost : ℕ := 16800000

/-- Theorem stating that the number of seasons is correct given the conditions. -/
theorem correct_number_of_seasons :
  (first_season_cost * first_season_episodes) +
  ((num_seasons - 2) * other_season_cost * middle_season_episodes) +
  (other_season_cost * last_season_episodes) = total_cost :=
sorry

end NUMINAMATH_CALUDE_correct_number_of_seasons_l4086_408684


namespace NUMINAMATH_CALUDE_remainder_problem_l4086_408698

theorem remainder_problem (d : ℕ) (h1 : d = 170) (h2 : d ∣ (690 - 10)) (h3 : ∃ r, d ∣ (875 - r)) :
  875 % d = 25 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l4086_408698


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l4086_408686

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 2 →
  Complex.abs b = Real.sqrt 26 →
  a * b = t - 2 * Complex.I →
  t > 0 →
  t = 10 := by
sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l4086_408686


namespace NUMINAMATH_CALUDE_base8_54321_equals_22737_l4086_408664

-- Define a function to convert a base-8 number to base-10
def base8ToBase10 (n : List Nat) : Nat :=
  List.foldl (fun acc d => 8 * acc + d) 0 n

-- Define the base-8 number 54321
def base8Number : List Nat := [5, 4, 3, 2, 1]

-- State the theorem
theorem base8_54321_equals_22737 :
  base8ToBase10 base8Number = 22737 := by
  sorry

end NUMINAMATH_CALUDE_base8_54321_equals_22737_l4086_408664


namespace NUMINAMATH_CALUDE_similar_right_triangles_perimeter_l4086_408626

theorem similar_right_triangles_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * a + a * a = b * b →  -- First triangle is right-angled with equal legs
  (c / b) * (c / b) = 2 →  -- Ratio of hypotenuses squared
  2 * ((c / b) * a) + c = 30 * Real.sqrt 2 + 30 := by
sorry

end NUMINAMATH_CALUDE_similar_right_triangles_perimeter_l4086_408626


namespace NUMINAMATH_CALUDE_edge_count_of_specific_polyhedron_l4086_408609

/-- A simple polyhedron is a polyhedron where each edge connects exactly two vertices and is part of exactly two faces. -/
structure SimplePolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- Euler's formula for simple polyhedra: F + V = E + 2 -/
axiom eulers_formula (p : SimplePolyhedron) : p.faces + p.vertices = p.edges + 2

theorem edge_count_of_specific_polyhedron :
  ∃ (p : SimplePolyhedron), p.faces = 12 ∧ p.vertices = 20 ∧ p.edges = 30 := by
  sorry

end NUMINAMATH_CALUDE_edge_count_of_specific_polyhedron_l4086_408609


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l4086_408632

theorem rectangle_area_increase (original_area : ℝ) (length_increase : ℝ) (width_increase : ℝ) : 
  original_area = 450 →
  length_increase = 0.2 →
  width_increase = 0.3 →
  original_area * (1 + length_increase) * (1 + width_increase) = 702 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l4086_408632
