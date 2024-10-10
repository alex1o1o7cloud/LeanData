import Mathlib

namespace regression_independence_correct_statement_l2489_248980

/-- Definition of regression analysis -/
def regression_analysis : Type := Unit

/-- Definition of independence test -/
def independence_test : Type := Unit

/-- Property: Regression analysis studies correlation between two variables -/
axiom regression_studies_correlation : regression_analysis → Prop

/-- Property: Independence test analyzes relationship between two variables -/
axiom independence_analyzes_relationship : independence_test → Prop

/-- Property: Independence test cannot determine relationships with 100% certainty -/
axiom independence_not_certain : independence_test → Prop

/-- The correct statement about regression analysis and independence test -/
def correct_statement : Prop :=
  ∃ (ra : regression_analysis) (it : independence_test),
    regression_studies_correlation ra ∧
    independence_analyzes_relationship it

theorem regression_independence_correct_statement :
  correct_statement :=
sorry

end regression_independence_correct_statement_l2489_248980


namespace complex_equation_sum_l2489_248965

theorem complex_equation_sum (a b : ℝ) : 
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a + b = 1 := by
  sorry

end complex_equation_sum_l2489_248965


namespace total_meals_is_48_l2489_248962

/-- Represents the number of entree options --/
def num_entrees : ℕ := 4

/-- Represents the number of drink options --/
def num_drinks : ℕ := 4

/-- Represents the number of dessert options (including "no dessert") --/
def num_desserts : ℕ := 3

/-- Calculates the total number of possible meal combinations --/
def total_meals : ℕ := num_entrees * num_drinks * num_desserts

/-- Theorem stating that the total number of possible meals is 48 --/
theorem total_meals_is_48 : total_meals = 48 := by
  sorry

end total_meals_is_48_l2489_248962


namespace sum_first_44_is_116_l2489_248921

/-- Represents the sequence where the nth 1 is followed by n 3s -/
def specialSequence (n : ℕ) : ℕ → ℕ
| 0 => 1
| k + 1 => if k < (n * (n + 1)) / 2 then
             if k = (n * (n - 1)) / 2 then 1 else 3
           else specialSequence (n + 1) k

/-- The sum of the first 44 terms of the special sequence -/
def sumFirst44 : ℕ := (List.range 44).map (specialSequence 1) |>.sum

/-- Theorem stating that the sum of the first 44 terms is 116 -/
theorem sum_first_44_is_116 : sumFirst44 = 116 := by sorry

end sum_first_44_is_116_l2489_248921


namespace ab_value_l2489_248905

theorem ab_value (a b : ℝ) (h : |3*a - 1| + b^2 = 0) : a^b = 1 := by
  sorry

end ab_value_l2489_248905


namespace max_class_size_is_17_l2489_248975

/-- Represents a school with students and buses for an excursion. -/
structure School where
  total_students : ℕ
  num_buses : ℕ
  seats_per_bus : ℕ

/-- Checks if it's possible to seat all students with the given maximum class size. -/
def can_seat_all (s : School) (max_class_size : ℕ) : Prop :=
  ∀ (class_sizes : List ℕ),
    (class_sizes.sum = s.total_students) →
    (∀ size ∈ class_sizes, size ≤ max_class_size) →
    ∃ (allocation : List (List ℕ)),
      (allocation.length ≤ s.num_buses) ∧
      (∀ bus ∈ allocation, bus.sum ≤ s.seats_per_bus) ∧
      (allocation.join.sum = s.total_students)

/-- The main theorem stating that 17 is the maximum class size for the given school configuration. -/
theorem max_class_size_is_17 (s : School)
    (h1 : s.total_students = 920)
    (h2 : s.num_buses = 16)
    (h3 : s.seats_per_bus = 71) :
    (can_seat_all s 17 ∧ ¬can_seat_all s 18) :=
  sorry


end max_class_size_is_17_l2489_248975


namespace binomial_coefficient_sum_l2489_248986

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) : 
  (∀ x : ℤ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end binomial_coefficient_sum_l2489_248986


namespace jake_flower_charge_l2489_248912

/-- The amount Jake should charge for planting flowers -/
def flower_charge (mowing_rate : ℚ) (desired_rate : ℚ) (mowing_time : ℚ) (planting_time : ℚ) : ℚ :=
  planting_time * desired_rate + (desired_rate - mowing_rate) * mowing_time

/-- Theorem: Jake should charge $45 for planting flowers -/
theorem jake_flower_charge :
  flower_charge 15 20 1 2 = 45 := by
  sorry

end jake_flower_charge_l2489_248912


namespace circle_intersects_lines_iff_radius_in_range_l2489_248992

/-- A circle in a 2D Cartesian coordinate system. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a point is on a circle. -/
def on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Predicate to check if a point is at distance 1 from x-axis. -/
def dist_1_from_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 = 1 ∨ p.2 = -1

/-- The main theorem statement. -/
theorem circle_intersects_lines_iff_radius_in_range (r : ℝ) :
  (∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    on_circle { center := (3, -5), radius := r } p1 ∧
    on_circle { center := (3, -5), radius := r } p2 ∧
    dist_1_from_x_axis p1 ∧
    dist_1_from_x_axis p2) ↔
  (4 < r ∧ r < 6) :=
sorry

end circle_intersects_lines_iff_radius_in_range_l2489_248992


namespace smallest_reducible_fraction_l2489_248978

theorem smallest_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (n - 17 : ℤ) ≠ 0 ∧
  (7 * n + 2 : ℤ) ≠ 0 ∧
  (∃ (k : ℤ), k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (7 * n + 2)) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n →
    (m - 17 : ℤ) = 0 ∨
    (7 * m + 2 : ℤ) = 0 ∨
    (∀ (k : ℤ), k > 1 → ¬(k ∣ (m - 17) ∧ k ∣ (7 * m + 2)))) ∧
  n = 28 :=
by sorry

end smallest_reducible_fraction_l2489_248978


namespace no_complex_root_for_integer_polynomial_l2489_248988

/-- A polynomial of degree 4 with leading coefficient 1 and integer coefficients -/
def IntegerPolynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℤ, ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + d

/-- The property that a polynomial has two integer roots -/
def HasTwoIntegerRoots (P : ℝ → ℝ) : Prop :=
  ∃ p q : ℤ, P p = 0 ∧ P q = 0

/-- Complex number of the form (a + b*i)/2 where a and b are integers and b is non-zero -/
def ComplexRoot (z : ℂ) : Prop :=
  ∃ a b : ℤ, z = (a + b*Complex.I)/2 ∧ b ≠ 0

theorem no_complex_root_for_integer_polynomial (P : ℝ → ℝ) :
  IntegerPolynomial P → HasTwoIntegerRoots P →
  ¬∃ z : ℂ, ComplexRoot z ∧ (P z.re = 0 ∧ P z.im = 0) :=
sorry

end no_complex_root_for_integer_polynomial_l2489_248988


namespace distance_to_origin_of_complex_fraction_l2489_248958

theorem distance_to_origin_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end distance_to_origin_of_complex_fraction_l2489_248958


namespace girls_to_boys_ratio_l2489_248900

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) (girls boys : ℕ) : 
  total = 30 →
  difference = 6 →
  girls = boys + difference →
  total = girls + boys →
  (girls : ℚ) / (boys : ℚ) = 3 / 2 :=
by
  sorry

end girls_to_boys_ratio_l2489_248900


namespace complex_modulus_problem_l2489_248972

theorem complex_modulus_problem (z : ℂ) (x : ℝ) 
  (h1 : z * Complex.I = 2 * Complex.I + x)
  (h2 : z.im = 2) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l2489_248972


namespace sector_central_angle_l2489_248995

theorem sector_central_angle (r : ℝ) (θ : ℝ) (h : r > 0) :
  2 * r + r * θ = π * r / 2 → θ = π - 2 := by
  sorry

end sector_central_angle_l2489_248995


namespace rationalize_denominator_l2489_248911

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end rationalize_denominator_l2489_248911


namespace teacher_distribution_count_l2489_248987

/-- The number of ways to distribute 4 teachers among 3 middle schools -/
def distribute_teachers : ℕ :=
  Nat.choose 4 2 * Nat.factorial 3

/-- Theorem: The number of ways to distribute 4 teachers among 3 middle schools,
    with each school having at least one teacher, is equal to 36 -/
theorem teacher_distribution_count : distribute_teachers = 36 := by
  sorry

end teacher_distribution_count_l2489_248987


namespace pascals_identity_l2489_248993

theorem pascals_identity (n k : ℕ) : 
  Nat.choose n k + Nat.choose n (k + 1) = Nat.choose (n + 1) (k + 1) := by
  sorry

end pascals_identity_l2489_248993


namespace total_cost_approx_l2489_248999

/-- Calculate the final price of an item after discounts and tax -/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (taxRate : ℝ) : ℝ :=
  let priceAfterDiscount1 := originalPrice * (1 - discount1)
  let priceAfterDiscount2 := priceAfterDiscount1 * (1 - discount2)
  let taxAmount := priceAfterDiscount2 * taxRate
  priceAfterDiscount2 + taxAmount

/-- Calculate the total cost of all items -/
def totalCost (item1Price : ℝ) (item2Price : ℝ) (item3Price : ℝ) : ℝ :=
  let item1 := finalPrice item1Price 0.25 0.15 0.07
  let item2 := finalPrice item2Price 0.30 0 0.10
  let item3 := finalPrice item3Price 0.20 0 0.05
  item1 + item2 + item3

/-- Theorem: The total cost for all three items is approximately $335.93 -/
theorem total_cost_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ abs (totalCost 200 150 100 - 335.93) < ε :=
sorry

end total_cost_approx_l2489_248999


namespace drugstore_inventory_theorem_l2489_248945

def bottles_delivered (initial_inventory : ℕ) (monday_sales : ℕ) (tuesday_sales : ℕ) (daily_sales_wed_to_sun : ℕ) (final_inventory : ℕ) : ℕ :=
  let total_sales := monday_sales + tuesday_sales + (daily_sales_wed_to_sun * 5)
  let remaining_before_delivery := initial_inventory - (monday_sales + tuesday_sales + (daily_sales_wed_to_sun * 4))
  final_inventory - remaining_before_delivery

theorem drugstore_inventory_theorem (initial_inventory : ℕ) (monday_sales : ℕ) (tuesday_sales : ℕ) (daily_sales_wed_to_sun : ℕ) (final_inventory : ℕ) 
  (h1 : initial_inventory = 4500)
  (h2 : monday_sales = 2445)
  (h3 : tuesday_sales = 900)
  (h4 : daily_sales_wed_to_sun = 50)
  (h5 : final_inventory = 1555) :
  bottles_delivered initial_inventory monday_sales tuesday_sales daily_sales_wed_to_sun final_inventory = 600 := by
  sorry

end drugstore_inventory_theorem_l2489_248945


namespace square_less_than_self_for_unit_interval_l2489_248944

theorem square_less_than_self_for_unit_interval (x : ℝ) : 0 < x → x < 1 → x^2 < x := by
  sorry

end square_less_than_self_for_unit_interval_l2489_248944


namespace factorizations_of_2079_l2489_248907

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ a * b = 2079

def distinct_factorizations (f1 f2 : ℕ × ℕ) : Prop :=
  f1.1 ≠ f2.1 ∧ f1.1 ≠ f2.2

theorem factorizations_of_2079 :
  ∃ (f1 f2 : ℕ × ℕ),
    valid_factorization f1.1 f1.2 ∧
    valid_factorization f2.1 f2.2 ∧
    distinct_factorizations f1 f2 ∧
    ∀ (f : ℕ × ℕ), valid_factorization f.1 f.2 →
      (f = f1 ∨ f = f2 ∨ f = (f1.2, f1.1) ∨ f = (f2.2, f2.1)) :=
sorry

end factorizations_of_2079_l2489_248907


namespace total_cans_in_both_closets_l2489_248994

/-- Represents the capacity of a closet for storing cans -/
structure ClosetCapacity where
  cansPerRow : Nat
  rowsPerShelf : Nat
  shelves : Nat

/-- Calculates the total number of cans that can be stored in a closet -/
def totalCansInCloset (capacity : ClosetCapacity) : Nat :=
  capacity.cansPerRow * capacity.rowsPerShelf * capacity.shelves

/-- The capacity of the first closet -/
def firstCloset : ClosetCapacity :=
  { cansPerRow := 12, rowsPerShelf := 4, shelves := 10 }

/-- The capacity of the second closet -/
def secondCloset : ClosetCapacity :=
  { cansPerRow := 15, rowsPerShelf := 5, shelves := 8 }

/-- Theorem stating the total number of cans Jack can store in both closets -/
theorem total_cans_in_both_closets :
  totalCansInCloset firstCloset + totalCansInCloset secondCloset = 1080 := by
  sorry

end total_cans_in_both_closets_l2489_248994


namespace power_of_two_equality_l2489_248902

theorem power_of_two_equality (x : ℕ) : (1 / 16 : ℚ) * 2^50 = 2^x → x = 46 := by
  sorry

end power_of_two_equality_l2489_248902


namespace largest_product_of_three_l2489_248929

def S : Finset Int := {-5, -4, -1, 3, 7, 9}

theorem largest_product_of_three (a b c : Int) :
  a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S →
  x ≠ y → y ≠ z → x ≠ z →
  a * b * c ≤ x * y * z →
  x * y * z ≤ 189 :=
by sorry

end largest_product_of_three_l2489_248929


namespace right_triangle_identification_l2489_248956

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_identification :
  (is_right_triangle 1 (Real.sqrt 2) (Real.sqrt 3)) ∧
  ¬(is_right_triangle (Real.sqrt 2) (Real.sqrt 3) 2) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 9 16 25) :=
by sorry

end right_triangle_identification_l2489_248956


namespace quadratic_coefficient_values_l2489_248916

/-- Given an algebraic expression x^2 + px + q, prove that p = 0 and q = -6
    when the expression equals -5 for x = -1 and 3 for x = 3. -/
theorem quadratic_coefficient_values (p q : ℝ) : 
  ((-1)^2 + p*(-1) + q = -5) ∧ (3^2 + p*3 + q = 3) → p = 0 ∧ q = -6 := by
  sorry

end quadratic_coefficient_values_l2489_248916


namespace sweater_markup_l2489_248901

theorem sweater_markup (wholesale_price : ℝ) (h1 : wholesale_price > 0) :
  let discounted_price := 1.4 * wholesale_price
  let retail_price := 2 * discounted_price
  let markup := (retail_price - wholesale_price) / wholesale_price * 100
  markup = 180 := by
  sorry

end sweater_markup_l2489_248901


namespace tonya_final_stamps_l2489_248926

/-- The number of matches equivalent to one stamp -/
def matches_per_stamp : ℕ := 12

/-- The number of matches in each matchbook -/
def matches_per_matchbook : ℕ := 24

/-- The number of stamps Tonya initially has -/
def tonya_initial_stamps : ℕ := 13

/-- The number of matchbooks Jimmy has -/
def jimmy_matchbooks : ℕ := 5

/-- Calculate the number of stamps Tonya has left after trading with Jimmy -/
def tonya_stamps_left : ℕ := 
  tonya_initial_stamps - (jimmy_matchbooks * matches_per_matchbook) / matches_per_stamp

theorem tonya_final_stamps : tonya_stamps_left = 3 := by
  sorry

end tonya_final_stamps_l2489_248926


namespace smallest_right_triangle_area_l2489_248991

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 5) :
  let c := Real.sqrt (a^2 - b^2)
  (5 * Real.sqrt 11) / 2 = min ((a * b) / 2) ((b * c) / 2) := by
  sorry

end smallest_right_triangle_area_l2489_248991


namespace prob_at_least_one_of_three_l2489_248982

/-- The probability that at least one of three independent events occurs -/
theorem prob_at_least_one_of_three (p₁ p₂ p₃ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) 
  (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) 
  (h₃ : 0 ≤ p₃ ∧ p₃ ≤ 1) : 
  1 - (1 - p₁) * (1 - p₂) * (1 - p₃) = 
  1 - ((1 - p₁) * (1 - p₂) * (1 - p₃)) :=
by sorry

end prob_at_least_one_of_three_l2489_248982


namespace socorro_training_days_l2489_248981

/-- Calculates the number of days required to complete a training program. -/
def trainingDays (totalHours : ℕ) (dailyMinutes : ℕ) : ℕ :=
  (totalHours * 60) / dailyMinutes

/-- Proves that given 5 hours of total training time and 30 minutes of daily training,
    it takes 10 days to complete the training. -/
theorem socorro_training_days :
  trainingDays 5 30 = 10 := by
  sorry

end socorro_training_days_l2489_248981


namespace bird_count_problem_l2489_248917

/-- The number of grey birds initially in the cage -/
def initial_grey_birds : ℕ := 40

/-- The number of white birds next to the cage -/
def white_birds : ℕ := initial_grey_birds + 6

/-- The number of grey birds remaining in the cage after ten minutes -/
def remaining_grey_birds : ℕ := initial_grey_birds / 2

/-- The total number of birds remaining after ten minutes -/
def total_remaining_birds : ℕ := 66

theorem bird_count_problem :
  white_birds + remaining_grey_birds = total_remaining_birds :=
sorry

end bird_count_problem_l2489_248917


namespace inverse_prop_is_false_l2489_248909

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the interval [a,b]
variable (a b : ℝ)

-- State that f is continuous on [a,b]
variable (hf : ContinuousOn f (Set.Icc a b))

-- Define the original proposition
def original_prop : Prop :=
  ∀ x ∈ Set.Ioo a b, f a * f b < 0 → ∃ c ∈ Set.Ioo a b, f c = 0

-- Define the inverse proposition
def inverse_prop : Prop :=
  ∀ x ∈ Set.Ioo a b, (∃ c ∈ Set.Ioo a b, f c = 0) → f a * f b < 0

-- State the theorem
theorem inverse_prop_is_false
  (h : original_prop f a b) : ¬(inverse_prop f a b) := by
  sorry


end inverse_prop_is_false_l2489_248909


namespace max_min_sum_zero_l2489_248919

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_min_sum_zero :
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∀ x, f x ≥ n) ∧ (m + n = 0) := by
  sorry

end max_min_sum_zero_l2489_248919


namespace triangle_property_l2489_248953

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.b * Real.tan t.B = Real.sqrt 3 * (t.a * Real.cos t.C + t.c * Real.cos t.A))
  (h2 : t.b = 2 * Real.sqrt 3)
  (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3) :
  t.B = π / 3 ∧ t.a + t.c = 4 * Real.sqrt 3 := by
  sorry

end triangle_property_l2489_248953


namespace correct_verb_forms_l2489_248997

-- Define the structure of a sentence
structure Sentence where
  subject : String
  verb1 : String
  verb2 : String

-- Define a predicate for plural subjects
def is_plural (s : String) : Prop := s.endsWith "s"

-- Define a predicate for partial references
def is_partial_reference (s : String) : Prop := s = "some"

-- Define a predicate for plural verb forms
def is_plural_verb (v : String) : Prop := v = "are" ∨ v = "seem"

-- Theorem statement
theorem correct_verb_forms (s : Sentence) 
  (h1 : is_plural s.subject) 
  (h2 : is_partial_reference "some") : 
  is_plural_verb s.verb1 ∧ is_plural_verb s.verb2 := by
  sorry

-- Example usage
def example_sentence : Sentence := {
  subject := "Such phenomena"
  verb1 := "are"
  verb2 := "seem"
}

#check correct_verb_forms example_sentence

end correct_verb_forms_l2489_248997


namespace sqrt_cubed_equals_64_l2489_248941

theorem sqrt_cubed_equals_64 (x : ℝ) : (Real.sqrt x)^3 = 64 → x = 16 := by
  sorry

end sqrt_cubed_equals_64_l2489_248941


namespace no_overlapping_attendees_l2489_248998

theorem no_overlapping_attendees (total_guests : ℕ) 
  (oates_attendees hall_attendees singh_attendees brown_attendees : ℕ) :
  total_guests = 350 ∧
  oates_attendees = 105 ∧
  hall_attendees = 98 ∧
  singh_attendees = 82 ∧
  brown_attendees = 65 ∧
  oates_attendees + hall_attendees + singh_attendees + brown_attendees = total_guests →
  (∃ (overlapping_attendees : ℕ), overlapping_attendees = 0) :=
by sorry

end no_overlapping_attendees_l2489_248998


namespace polygon_interior_angle_sum_l2489_248946

theorem polygon_interior_angle_sum (n : ℕ) (h : n > 2) :
  (360 / 72 : ℝ) = n →
  (n - 2) * 180 = 540 :=
by sorry

end polygon_interior_angle_sum_l2489_248946


namespace line_points_property_l2489_248984

theorem line_points_property (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -2 * x₁ + 3)
  (h2 : y₂ = -2 * x₂ + 3)
  (h3 : y₃ = -2 * x₃ + 3)
  (h4 : x₁ < x₂)
  (h5 : x₂ < x₃)
  (h6 : x₂ * x₃ < 0) :
  y₁ * y₂ > 0 := by
  sorry

end line_points_property_l2489_248984


namespace range_of_b_l2489_248990

/-- The function f(x) = x² + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The set A of zeros of f -/
def A (b c : ℝ) : Set ℝ := {x | f b c x = 0}

/-- The set B of x such that f(f(x)) = 0 -/
def B (b c : ℝ) : Set ℝ := {x | f b c (f b c x) = 0}

/-- If there exists x₀ in B but not in A, then b < 0 or b ≥ 4 -/
theorem range_of_b (b c : ℝ) : 
  (∃ x₀, x₀ ∈ B b c ∧ x₀ ∉ A b c) → b < 0 ∨ b ≥ 4 := by
  sorry

end range_of_b_l2489_248990


namespace non_negative_y_range_l2489_248954

theorem non_negative_y_range (x : Real) :
  0 ≤ x ∧ x ≤ Real.pi / 2 →
  (∃ y : Real, y = 4 * Real.cos x * Real.sin x + 2 * Real.cos x - 2 * Real.sin x - 1 ∧ y ≥ 0) ↔
  0 ≤ x ∧ x ≤ Real.pi / 3 := by
sorry

end non_negative_y_range_l2489_248954


namespace max_expensive_price_theorem_l2489_248996

/-- Represents a set of products with their prices -/
structure ProductSet where
  prices : Finset ℝ
  count : Nat
  avg_price : ℝ
  min_price : ℝ
  low_price_count : Nat
  low_price_threshold : ℝ

/-- The maximum possible price for the most expensive product -/
def max_expensive_price (ps : ProductSet) : ℝ :=
  ps.count * ps.avg_price
    - (ps.low_price_count * ps.min_price
      + (ps.count - ps.low_price_count - 1) * ps.low_price_threshold)

/-- Theorem stating the maximum price of the most expensive product -/
theorem max_expensive_price_theorem (ps : ProductSet)
  (h_count : ps.count = 25)
  (h_avg_price : ps.avg_price = 1200)
  (h_min_price : ps.min_price = 400)
  (h_low_price_count : ps.low_price_count = 12)
  (h_low_price_threshold : ps.low_price_threshold = 1000)
  (h_prices_above_min : ∀ p ∈ ps.prices, p ≥ ps.min_price)
  (h_low_price_count_correct : (ps.prices.filter (· < ps.low_price_threshold)).card = ps.low_price_count) :
  max_expensive_price ps = 13200 := by
  sorry

end max_expensive_price_theorem_l2489_248996


namespace scientific_notation_equality_l2489_248903

/-- The number to be expressed in scientific notation -/
def original_number : ℝ := 384000000

/-- The coefficient in scientific notation -/
def coefficient : ℝ := 3.84

/-- The exponent in scientific notation -/
def exponent : ℕ := 8

/-- Theorem stating that the original number is equal to its scientific notation form -/
theorem scientific_notation_equality :
  original_number = coefficient * (10 : ℝ) ^ exponent := by sorry

end scientific_notation_equality_l2489_248903


namespace division_problem_l2489_248971

theorem division_problem (L S Q : ℕ) (h1 : L - S = 1365) (h2 : L = 1631) (h3 : L = S * Q + 35) : Q = 6 := by
  sorry

end division_problem_l2489_248971


namespace solution_set_of_inequality_l2489_248924

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x - x^2 > 0) ↔ (x > 0 ∧ x < 2) := by sorry

end solution_set_of_inequality_l2489_248924


namespace larger_number_proof_l2489_248937

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1311) (h3 : L = 11 * S + 11) : L = 1441 := by
  sorry

end larger_number_proof_l2489_248937


namespace tan_alpha_eq_two_l2489_248963

theorem tan_alpha_eq_two (α : ℝ) (h : 2 * Real.sin α + Real.cos α = -Real.sqrt 5) :
  Real.tan α = 2 := by
  sorry

end tan_alpha_eq_two_l2489_248963


namespace six_by_six_square_1x4_rectangles_impossible_l2489_248936

theorem six_by_six_square_1x4_rectangles_impossible : ¬ ∃ (a b : ℕ), 
  a + 4*b = 6 ∧ 4*a + b = 6 :=
sorry

end six_by_six_square_1x4_rectangles_impossible_l2489_248936


namespace largest_consecutive_odd_sum_55_l2489_248964

theorem largest_consecutive_odd_sum_55 :
  (∃ (n : ℕ) (x : ℕ),
    n > 0 ∧
    x > 0 ∧
    x % 2 = 1 ∧
    n * (x + n - 1) = 55 ∧
    ∀ (m : ℕ), m > n →
      ¬∃ (y : ℕ), y > 0 ∧ y % 2 = 1 ∧ m * (y + m - 1) = 55) →
  (∃ (x : ℕ),
    x > 0 ∧
    x % 2 = 1 ∧
    11 * (x + 11 - 1) = 55 ∧
    ∀ (m : ℕ), m > 11 →
      ¬∃ (y : ℕ), y > 0 ∧ y % 2 = 1 ∧ m * (y + m - 1) = 55) :=
by sorry

end largest_consecutive_odd_sum_55_l2489_248964


namespace nested_radical_value_l2489_248967

theorem nested_radical_value : 
  ∃ x : ℝ, x = Real.sqrt (2 + x) ∧ x ≥ 0 ∧ 2 + x ≥ 0 → x = 2 := by
  sorry

end nested_radical_value_l2489_248967


namespace projection_magnitude_l2489_248925

theorem projection_magnitude (a b : ℝ × ℝ) :
  (a.1 * b.1 + a.2 * b.2 = -2) →
  (b = (1, Real.sqrt 3)) →
  let c := ((a.1 * b.1 + a.2 * b.2) / (b.1 ^ 2 + b.2 ^ 2)) • b
  (b.1 - c.1) ^ 2 + (b.2 - c.2) ^ 2 = 9 := by
  sorry

end projection_magnitude_l2489_248925


namespace tangent_line_length_range_l2489_248935

-- Define the circles
def circle_C1 (x y α : ℝ) : Prop := (x + Real.cos α)^2 + (y + Real.sin α)^2 = 4
def circle_C2 (x y β : ℝ) : Prop := (x - 5 * Real.sin β)^2 + (y - 5 * Real.cos β)^2 = 1

-- Define the range of α and β
def angle_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the tangent line MN
def tangent_line (M N : ℝ × ℝ) (α β : ℝ) : Prop :=
  circle_C1 M.1 M.2 α ∧ circle_C2 N.1 N.2 β ∧
  ∃ (t : ℝ), N = (M.1 + t * (N.1 - M.1), M.2 + t * (N.2 - M.2))

-- State the theorem
theorem tangent_line_length_range :
  ∀ (M N : ℝ × ℝ) (α β : ℝ),
  angle_range α → angle_range β → tangent_line M N α β →
  2 * Real.sqrt 2 ≤ Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) ∧
  Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) ≤ 3 * Real.sqrt 7 :=
sorry

end tangent_line_length_range_l2489_248935


namespace probability_x_plus_y_even_l2489_248974

def X := Finset.range 5
def Y := Finset.range 4

theorem probability_x_plus_y_even :
  let total_outcomes := X.card * Y.card
  let favorable_outcomes := (X.filter (λ x => x % 2 = 0)).card * (Y.filter (λ y => y % 2 = 0)).card +
                            (X.filter (λ x => x % 2 = 1)).card * (Y.filter (λ y => y % 2 = 1)).card
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by
  sorry

end probability_x_plus_y_even_l2489_248974


namespace median_salary_is_clerk_salary_l2489_248947

/-- Represents a position in the company -/
inductive Position
  | CEO
  | SeniorManager
  | Manager
  | AssistantManager
  | Clerk

/-- Returns the number of employees for a given position -/
def employeeCount (p : Position) : ℕ :=
  match p with
  | .CEO => 1
  | .SeniorManager => 8
  | .Manager => 12
  | .AssistantManager => 10
  | .Clerk => 40

/-- Returns the salary for a given position -/
def salary (p : Position) : ℕ :=
  match p with
  | .CEO => 180000
  | .SeniorManager => 95000
  | .Manager => 70000
  | .AssistantManager => 55000
  | .Clerk => 28000

/-- The total number of employees in the company -/
def totalEmployees : ℕ := 71

/-- Theorem stating that the median salary is equal to the Clerk's salary -/
theorem median_salary_is_clerk_salary :
  (totalEmployees + 1) / 2 ≤ (employeeCount Position.Clerk) ∧
  (totalEmployees + 1) / 2 > (totalEmployees - employeeCount Position.Clerk) →
  salary Position.Clerk = 28000 := by
  sorry

#check median_salary_is_clerk_salary

end median_salary_is_clerk_salary_l2489_248947


namespace a_lt_c_lt_b_l2489_248910

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Conditions
axiom derivative_f : ∀ x, HasDerivAt f (f' x) x

axiom symmetry_f' : ∀ x, f' (x - 1) = f' (1 - x)

axiom symmetry_f : ∀ x, f x = f (2 - x)

axiom monotone_f : MonotoneOn f (Set.Icc (-7) (-6))

-- Define a, b, and c
def a : ℝ := f (Real.log (6 * Real.exp 1 / 5))
def b : ℝ := f (Real.exp 0.2 - 1)
def c : ℝ := f (2 / 9)

-- Theorem to prove
theorem a_lt_c_lt_b : a < c ∧ c < b :=
  sorry

end a_lt_c_lt_b_l2489_248910


namespace mara_marbles_l2489_248934

theorem mara_marbles (mara_bags : ℕ) (markus_bags : ℕ) (markus_marbles_per_bag : ℕ) :
  mara_bags = 12 →
  markus_bags = 2 →
  markus_marbles_per_bag = 13 →
  ∃ (mara_marbles_per_bag : ℕ),
    mara_bags * mara_marbles_per_bag + 2 = markus_bags * markus_marbles_per_bag ∧
    mara_marbles_per_bag = 2 :=
by sorry

end mara_marbles_l2489_248934


namespace product_of_4_7_25_l2489_248948

theorem product_of_4_7_25 : 4 * 7 * 25 = 700 := by
  sorry

end product_of_4_7_25_l2489_248948


namespace max_value_cube_roots_l2489_248906

theorem max_value_cube_roots (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) : 
  (a * b * c) ^ (1/3 : ℝ) + ((2 - a) * (2 - b) * (2 - c)) ^ (1/3 : ℝ) ≤ 2 ∧ 
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ 
    (x * x * x) ^ (1/3 : ℝ) + ((2 - x) * (2 - x) * (2 - x)) ^ (1/3 : ℝ) = 2 :=
by sorry

end max_value_cube_roots_l2489_248906


namespace white_paper_bunches_l2489_248989

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The number of sheets in a bunch -/
def sheets_per_bunch : ℕ := 4

/-- The number of sheets in a bundle -/
def sheets_per_bundle : ℕ := 2

/-- The number of sheets in a heap -/
def sheets_per_heap : ℕ := 20

/-- The total number of sheets removed -/
def total_sheets_removed : ℕ := 114

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

theorem white_paper_bunches :
  white_bunches * sheets_per_bunch = 
    total_sheets_removed - 
    (colored_bundles * sheets_per_bundle + scrap_heaps * sheets_per_heap) :=
by sorry

end white_paper_bunches_l2489_248989


namespace value_of_x_minus_4y_l2489_248960

theorem value_of_x_minus_4y (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - 3 * y = 10) :
  x - 4 * y = 5 := by
  sorry

end value_of_x_minus_4y_l2489_248960


namespace mans_rowing_speed_l2489_248955

/-- The speed of a man rowing in still water, given his downstream speed and current speed. -/
theorem mans_rowing_speed (downstream_distance : ℝ) (downstream_time : ℝ) (current_speed : ℝ) : 
  downstream_distance / downstream_time * 3600 / 1000 - current_speed = 6 :=
by
  sorry

#check mans_rowing_speed 110 44 3

end mans_rowing_speed_l2489_248955


namespace problem_1_problem_2_l2489_248959

-- Define the mixed number addition function
def mixed_number_add (a b c d : ℚ) : ℚ := a + b + c + d

-- Theorem 1
theorem problem_1 : 
  mixed_number_add (-2020 - 2/3) (2019 + 3/4) (-2018 - 5/6) (2017 + 1/2) = -2 - 1/4 := by sorry

-- Theorem 2
theorem problem_2 : 
  mixed_number_add (-1 - 1/2) (-2000 - 5/6) (4000 + 3/4) (-1999 - 2/3) = -5/4 := by sorry

end problem_1_problem_2_l2489_248959


namespace f_of_A_eq_l2489_248918

/-- The matrix A --/
def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 2, 3]

/-- The polynomial function f --/
def f (x : Matrix (Fin 2) (Fin 2) ℤ) : Matrix (Fin 2) (Fin 2) ℤ := x^2 - 5 • x

/-- Theorem stating that f(A) equals the given result --/
theorem f_of_A_eq : f A = !![(-6), 1; (-2), (-8)] := by sorry

end f_of_A_eq_l2489_248918


namespace prob_red_after_transfer_l2489_248942

-- Define the initial contents of bags A and B
def bag_A : Finset (Fin 3) := {0, 1, 2}
def bag_B : Finset (Fin 3) := {0, 1, 2}

-- Define the number of balls of each color in bag A
def red_A : ℕ := 3
def white_A : ℕ := 2
def black_A : ℕ := 5

-- Define the number of balls of each color in bag B
def red_B : ℕ := 3
def white_B : ℕ := 3
def black_B : ℕ := 4

-- Define the total number of balls in each bag
def total_A : ℕ := red_A + white_A + black_A
def total_B : ℕ := red_B + white_B + black_B

-- Define the probability of drawing a red ball from bag B after transfer
def prob_red_B : ℚ := 3 / 10

-- State the theorem
theorem prob_red_after_transfer : 
  (red_A * (red_B + 1) + white_A * red_B + black_A * red_B) / 
  (total_A * (total_B + 1)) = prob_red_B := by sorry

end prob_red_after_transfer_l2489_248942


namespace quadratic_equations_properties_l2489_248968

/-- The quadratic equation x^2 + mx + 1 = 0 has two distinct negative real roots -/
def has_two_distinct_negative_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- The quadratic equation 4x^2 + (4m-2)x + 1 = 0 does not have any real roots -/
def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + (4*m-2)*x + 1 ≠ 0

theorem quadratic_equations_properties (m : ℝ) :
  (has_no_real_roots m ↔ 1/2 < m ∧ m < 3/2) ∧
  (has_two_distinct_negative_roots m ∧ ¬has_no_real_roots m ↔ m > 2) := by
  sorry

end quadratic_equations_properties_l2489_248968


namespace mixed_doubles_selection_count_l2489_248979

theorem mixed_doubles_selection_count :
  let male_count : ℕ := 5
  let female_count : ℕ := 4
  male_count * female_count = 20 :=
by sorry

end mixed_doubles_selection_count_l2489_248979


namespace distribute_six_balls_four_boxes_l2489_248969

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 187 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem distribute_six_balls_four_boxes : distribute_balls 6 4 = 187 := by
  sorry

end distribute_six_balls_four_boxes_l2489_248969


namespace system_solution_unique_l2489_248985

theorem system_solution_unique (x y : ℝ) : 
  (x + 3 * y = 2 ∧ 4 * x - y = 8) ↔ (x = 2 ∧ y = 0) := by
sorry

end system_solution_unique_l2489_248985


namespace mixture_cost_july_l2489_248976

/-- The cost of a mixture of milk powder and coffee in July -/
def mixture_cost (june_cost : ℝ) : ℝ :=
  let july_coffee_cost := june_cost * 4
  let july_milk_cost := june_cost * 0.2
  (1.5 * july_coffee_cost) + (1.5 * july_milk_cost)

/-- Theorem: The cost of a 3 lbs mixture of equal parts milk powder and coffee in July is $6.30 -/
theorem mixture_cost_july : ∃ (june_cost : ℝ), 
  (june_cost * 0.2 = 0.20) ∧ (mixture_cost june_cost = 6.30) := by
  sorry

end mixture_cost_july_l2489_248976


namespace complex_fraction_power_2000_l2489_248931

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_power_2000 : ((1 - i) / (1 + i)) ^ 2000 = 1 := by
  sorry

end complex_fraction_power_2000_l2489_248931


namespace symmetric_point_about_x_axis_l2489_248951

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of symmetry about the x-axis --/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The theorem to prove --/
theorem symmetric_point_about_x_axis :
  let A : Point := ⟨2, 1⟩
  let B : Point := ⟨2, -1⟩
  symmetricAboutXAxis A B := by sorry

end symmetric_point_about_x_axis_l2489_248951


namespace alex_not_reading_probability_l2489_248915

theorem alex_not_reading_probability (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end alex_not_reading_probability_l2489_248915


namespace walk_distance_proof_l2489_248923

/-- The distance Rajesh and Hiro walked together -/
def distance_together : ℝ := 7

/-- Hiro's walking distance -/
def hiro_distance : ℝ := distance_together

/-- Rajesh's walking distance -/
def rajesh_distance : ℝ := 18

theorem walk_distance_proof :
  (4 * hiro_distance - 10 = rajesh_distance) →
  distance_together = 7 := by
  sorry

end walk_distance_proof_l2489_248923


namespace stratified_sampling_proof_l2489_248961

/-- Represents a sampling method -/
inductive SamplingMethod
| Stratified
| Simple
| Cluster
| Systematic

/-- Represents the student population -/
structure Population where
  total : Nat
  male : Nat
  female : Nat

/-- Represents the sample -/
structure Sample where
  total : Nat
  male : Nat
  female : Nat

def is_stratified (pop : Population) (sam : Sample) : Prop :=
  (pop.male : Real) / pop.total = (sam.male : Real) / sam.total ∧
  (pop.female : Real) / pop.total = (sam.female : Real) / sam.total

theorem stratified_sampling_proof 
  (pop : Population) 
  (sam : Sample) 
  (h1 : pop.total = 1000) 
  (h2 : pop.male = 400) 
  (h3 : pop.female = 600) 
  (h4 : sam.total = 100) 
  (h5 : sam.male = 40) 
  (h6 : sam.female = 60) 
  (h7 : is_stratified pop sam) : 
  SamplingMethod.Stratified = SamplingMethod.Stratified :=
sorry

end stratified_sampling_proof_l2489_248961


namespace circular_film_radius_l2489_248928

/-- The radius of a circular film formed by pouring a liquid from a rectangular tank onto water -/
theorem circular_film_radius (tank_length tank_width tank_height film_thickness : ℝ)
  (tank_length_pos : tank_length > 0)
  (tank_width_pos : tank_width > 0)
  (tank_height_pos : tank_height > 0)
  (film_thickness_pos : film_thickness > 0)
  (h_tank_length : tank_length = 8)
  (h_tank_width : tank_width = 4)
  (h_tank_height : tank_height = 10)
  (h_film_thickness : film_thickness = 0.2) :
  let tank_volume := tank_length * tank_width * tank_height
  let film_radius := Real.sqrt (tank_volume / (π * film_thickness))
  film_radius = Real.sqrt (1600 / π) :=
by sorry

end circular_film_radius_l2489_248928


namespace conclusion_one_conclusion_three_l2489_248920

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a * (1 - b)

-- Theorem for the first correct conclusion
theorem conclusion_one : custom_op 2 (-2) = 6 := by sorry

-- Theorem for the third correct conclusion
theorem conclusion_three (a b : ℝ) (h : a + b = 0) :
  custom_op a a + custom_op b b = 2 * a * b := by sorry

end conclusion_one_conclusion_three_l2489_248920


namespace total_over_budget_l2489_248943

def project_budget (project : Char) : ℕ :=
  match project with
  | 'A' => 150000
  | 'B' => 120000
  | 'C' => 80000
  | _ => 0

def allocation_count (project : Char) : ℕ :=
  match project with
  | 'A' => 10
  | 'B' => 6
  | 'C' => 18
  | _ => 0

def allocation_period (project : Char) : ℕ :=
  match project with
  | 'A' => 2
  | 'B' => 3
  | 'C' => 1
  | _ => 0

def actual_spent (project : Char) : ℕ :=
  match project with
  | 'A' => 98450
  | 'B' => 72230
  | 'C' => 43065
  | _ => 0

def months_passed : ℕ := 9

def expected_expenditure (project : Char) : ℚ :=
  (project_budget project : ℚ) / (allocation_count project : ℚ) *
  ((months_passed : ℚ) / (allocation_period project : ℚ)).floor

def project_difference (project : Char) : ℚ :=
  (actual_spent project : ℚ) - expected_expenditure project

theorem total_over_budget :
  (project_difference 'A' + project_difference 'B' + project_difference 'C') = 38745 := by
  sorry

end total_over_budget_l2489_248943


namespace equation_solution_l2489_248977

theorem equation_solution : ∃! x : ℝ, 2 * x + 4 = |(-17 + 3)| :=
  by
    -- The unique solution is x = 5
    use 5
    -- Proof goes here
    sorry

end equation_solution_l2489_248977


namespace slope_of_line_from_equation_l2489_248983

theorem slope_of_line_from_equation (x y : ℝ) (h : (4 / x) + (5 / y) = 0) :
  ∃ m : ℝ, m = -5/4 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (4 / x₁ + 5 / y₁ = 0) → (4 / x₂ + 5 / y₂ = 0) → x₁ ≠ x₂ →
    (y₂ - y₁) / (x₂ - x₁) = m :=
by sorry

end slope_of_line_from_equation_l2489_248983


namespace jean_thursday_calls_correct_l2489_248930

/-- The number of calls Jean answered on each day of the week --/
structure CallData where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day --/
def average_calls : ℕ := 40

/-- The number of working days --/
def working_days : ℕ := 5

/-- Jean's actual call data for the week --/
def jean_calls : CallData where
  monday := 35
  tuesday := 46
  wednesday := 27
  thursday := 61  -- This is what we want to prove
  friday := 31

/-- Theorem stating that Jean's Thursday call count is correct --/
theorem jean_thursday_calls_correct :
  jean_calls.thursday = 
    working_days * average_calls - 
    (jean_calls.monday + jean_calls.tuesday + jean_calls.wednesday + jean_calls.friday) := by
  sorry


end jean_thursday_calls_correct_l2489_248930


namespace closest_perfect_square_to_320_l2489_248973

def closest_perfect_square (n : ℕ) : ℕ :=
  let root := n.sqrt
  if (root + 1)^2 - n < n - root^2
  then (root + 1)^2
  else root^2

theorem closest_perfect_square_to_320 :
  closest_perfect_square 320 = 324 :=
sorry

end closest_perfect_square_to_320_l2489_248973


namespace second_pipe_fill_time_l2489_248957

theorem second_pipe_fill_time (t1 t2 t3 t_all : ℝ) (h1 : t1 = 10) (h2 : t3 = 40) (h3 : t_all = 6.31578947368421) 
  (h4 : 1 / t1 + 1 / t2 - 1 / t3 = 1 / t_all) : t2 = 12 := by
  sorry

end second_pipe_fill_time_l2489_248957


namespace circle_center_on_line_max_ab_l2489_248939

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x - b*y + 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

theorem circle_center_on_line_max_ab :
  ∀ (a b : ℝ),
  line_equation a b (circle_center.1) (circle_center.2) →
  a * b ≤ 1/8 ∧
  ∀ (ε : ℝ), ε > 0 → ∃ (a' b' : ℝ), 
    line_equation a' b' (circle_center.1) (circle_center.2) ∧
    a' * b' > 1/8 - ε :=
by sorry

end circle_center_on_line_max_ab_l2489_248939


namespace xiao_ma_calculation_l2489_248949

theorem xiao_ma_calculation (x : ℤ) : 41 - x = 12 → 41 + x = 70 := by
  sorry

end xiao_ma_calculation_l2489_248949


namespace remainder_17_65_mod_7_l2489_248914

theorem remainder_17_65_mod_7 : 17^65 % 7 = 5 := by sorry

end remainder_17_65_mod_7_l2489_248914


namespace train_speed_in_kmh_l2489_248932

-- Define the length of the train in meters
def train_length : ℝ := 280

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 20

-- Define the conversion factor from m/s to km/h
def ms_to_kmh : ℝ := 3.6

-- Theorem to prove
theorem train_speed_in_kmh :
  (train_length / crossing_time) * ms_to_kmh = 50.4 := by
  sorry

end train_speed_in_kmh_l2489_248932


namespace subtracted_amount_l2489_248940

/-- Given a number N = 200, if 95% of N minus some amount A equals 178, then A must be 12. -/
theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 200 → 0.95 * N - A = 178 → A = 12 := by sorry

end subtracted_amount_l2489_248940


namespace triangle_height_l2489_248908

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 8 → area = 16 → area = (base * height) / 2 → height = 4 := by
  sorry

end triangle_height_l2489_248908


namespace complex_number_modulus_l2489_248927

theorem complex_number_modulus : 
  let z : ℂ := (4 - 2*I) / (1 + I)
  ‖z‖ = Real.sqrt 10 := by sorry

end complex_number_modulus_l2489_248927


namespace program_output_25_l2489_248913

theorem program_output_25 (x : ℝ) : 
  ((x < 0 ∧ (x + 1)^2 = 25) ∨ (x ≥ 0 ∧ (x - 1)^2 = 25)) ↔ (x = 6 ∨ x = -6) := by
  sorry

end program_output_25_l2489_248913


namespace transformed_quadratic_has_root_l2489_248950

/-- Given a quadratic polynomial with two roots, adding one root to the linear coefficient
    and subtracting its square from the constant term results in a polynomial with at least one root -/
theorem transformed_quadratic_has_root (a b r : ℝ) : 
  (∃ x y : ℝ, x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0 ∧ x ≠ y) →
  (∃ z : ℝ, z^2 + (a + r)*z + (b - r^2) = 0) ∧ 
  (r^2 + a*r + b = 0) :=
sorry

end transformed_quadratic_has_root_l2489_248950


namespace smallest_percent_increase_l2489_248966

def question_values : List ℕ := [100, 300, 600, 1000, 1500, 2500, 4000, 6500]

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def consecutive_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  List.zip l (List.tail l)

theorem smallest_percent_increase :
  let pairs := consecutive_pairs question_values
  let increases := List.map (fun (p : ℕ × ℕ) => percent_increase p.1 p.2) pairs
  List.argmin id increases = some 3 := by sorry

end smallest_percent_increase_l2489_248966


namespace unique_solution_is_two_l2489_248904

theorem unique_solution_is_two : 
  ∃! (x : ℝ), x > 0 ∧ x^(2^2) = 2^(x^2) ∧ x = 2 := by sorry

end unique_solution_is_two_l2489_248904


namespace tangent_double_angle_subtraction_l2489_248970

theorem tangent_double_angle_subtraction (α β : ℝ) 
  (h1 : Real.tan (α - β) = 2/5) 
  (h2 : Real.tan β = 1/2) : 
  Real.tan (α - 2*β) = -1/12 := by
  sorry

end tangent_double_angle_subtraction_l2489_248970


namespace scientific_notation_19672_l2489_248922

theorem scientific_notation_19672 :
  19672 = 1.9672 * (10 ^ 4) := by
  sorry

end scientific_notation_19672_l2489_248922


namespace expand_polynomial_product_l2489_248933

theorem expand_polynomial_product : ∀ x : ℝ,
  (3 * x^2 - 2 * x + 4) * (-4 * x^2 + 3 * x - 6) =
  -12 * x^4 + 17 * x^3 - 40 * x^2 + 24 * x - 24 :=
by
  sorry

end expand_polynomial_product_l2489_248933


namespace max_value_of_expression_l2489_248938

theorem max_value_of_expression (a b : ℝ) (h : a^2 + b^2 = 9) :
  ∃ (max : ℝ), max = 5 ∧ ∀ (x y : ℝ), x^2 + y^2 = 9 → x * y - y + x ≤ max :=
sorry

end max_value_of_expression_l2489_248938


namespace volume_to_surface_area_ratio_l2489_248952

/-- Represents a cube arrangement with specific properties -/
structure CubeArrangement where
  num_cubes : ℕ
  central_cube_exposed_faces : ℕ
  surrounding_cubes_exposed_faces : ℕ
  extending_cube_exposed_faces : ℕ

/-- Calculate the volume of the cube arrangement -/
def volume (c : CubeArrangement) : ℕ :=
  c.num_cubes

/-- Calculate the surface area of the cube arrangement -/
def surface_area (c : CubeArrangement) : ℕ :=
  c.surrounding_cubes_exposed_faces * 5 + c.central_cube_exposed_faces + c.extending_cube_exposed_faces

/-- The specific cube arrangement described in the problem -/
def special_arrangement : CubeArrangement :=
  { num_cubes := 8,
    central_cube_exposed_faces := 1,
    surrounding_cubes_exposed_faces := 5,
    extending_cube_exposed_faces := 3 }

/-- Theorem stating the ratio of volume to surface area for the special arrangement -/
theorem volume_to_surface_area_ratio :
  (volume special_arrangement : ℚ) / (surface_area special_arrangement : ℚ) = 8 / 29 := by
  sorry


end volume_to_surface_area_ratio_l2489_248952
