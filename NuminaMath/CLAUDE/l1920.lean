import Mathlib

namespace red_balls_count_l1920_192092

/-- Given a set of balls where the ratio of red balls to white balls is 4:5,
    and there are 20 white balls, prove that the number of red balls is 16. -/
theorem red_balls_count (total : ℕ) (red : ℕ) (white : ℕ) 
    (h1 : total = red + white)
    (h2 : red * 5 = white * 4)
    (h3 : white = 20) : red = 16 := by
  sorry

end red_balls_count_l1920_192092


namespace unitedNations75thAnniversary_l1920_192085

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (advanceDay d n)

-- Define the founding day of the United Nations
def unitedNationsFoundingDay : DayOfWeek := DayOfWeek.Wednesday

-- Define the number of days to advance for the 75th anniversary
def daysToAdvance : Nat := 93

-- Theorem statement
theorem unitedNations75thAnniversary :
  advanceDay unitedNationsFoundingDay daysToAdvance = DayOfWeek.Friday :=
sorry

end unitedNations75thAnniversary_l1920_192085


namespace quadratic_function_unique_l1920_192058

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  equal_roots : ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r)
  derivative : ∀ x, HasDerivAt f (2 * x + 2) x

/-- The main theorem: if f satisfies the given conditions, then f(x) = x^2 + 2x + 1 -/
theorem quadratic_function_unique (qf : QuadraticFunction) :
  ∀ x, qf.f x = x^2 + 2*x + 1 := by
  sorry

end quadratic_function_unique_l1920_192058


namespace sarah_brother_apple_ratio_l1920_192003

def sarah_apples : ℕ := 45
def brother_apples : ℕ := 9

theorem sarah_brother_apple_ratio :
  sarah_apples / brother_apples = 5 :=
sorry

end sarah_brother_apple_ratio_l1920_192003


namespace total_time_is_34_hours_l1920_192031

/-- Calculates the total time spent on drawing and coloring pictures. -/
def total_time (num_pictures : ℕ) (draw_time : ℝ) (color_time_reduction : ℝ) : ℝ :=
  let color_time := draw_time * (1 - color_time_reduction)
  num_pictures * (draw_time + color_time)

/-- Proves that the total time spent on all pictures is 34 hours. -/
theorem total_time_is_34_hours :
  total_time 10 2 0.3 = 34 := by
  sorry

end total_time_is_34_hours_l1920_192031


namespace not_prime_2011_2111_plus_2500_l1920_192020

theorem not_prime_2011_2111_plus_2500 : ¬ Nat.Prime (2011 * 2111 + 2500) := by
  sorry

end not_prime_2011_2111_plus_2500_l1920_192020


namespace parabola_c_value_l1920_192088

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) : ℝ → ℝ := fun x ↦ 2 * x^2 + b * x + c

theorem parabola_c_value :
  ∀ b c : ℝ, 
  Parabola b c 2 = 12 ∧ 
  Parabola b c 4 = 44 →
  c = -4 := by
sorry

end parabola_c_value_l1920_192088


namespace jelly_beans_in_jar_X_l1920_192027

/-- The number of jelly beans in jar X -/
def jarX (total : ℕ) (y : ℕ) : ℕ := 3 * y - 400

/-- The total number of jelly beans in both jars -/
def totalBeans (x y : ℕ) : ℕ := x + y

theorem jelly_beans_in_jar_X :
  ∃ (y : ℕ), totalBeans (jarX 1200 y) y = 1200 ∧ jarX 1200 y = 800 := by
  sorry

end jelly_beans_in_jar_X_l1920_192027


namespace stratified_sample_theorem_l1920_192037

/-- Represents the number of people in each age group -/
structure AgeGroups where
  over40 : ℕ
  between30and40 : ℕ
  under30 : ℕ

/-- Represents the sample sizes for each age group -/
structure SampleSizes where
  over40 : ℕ
  between30and40 : ℕ
  under30 : ℕ

/-- Calculates the stratified sample sizes given the total population, total sample size, and age group sizes -/
def calculateStratifiedSample (totalPopulation : ℕ) (totalSampleSize : ℕ) (ageGroups : AgeGroups) : SampleSizes :=
  let ratio := totalSampleSize / totalPopulation
  { over40 := ageGroups.over40 * ratio,
    between30and40 := ageGroups.between30and40 * ratio,
    under30 := ageGroups.under30 * ratio }

theorem stratified_sample_theorem (totalPopulation : ℕ) (totalSampleSize : ℕ) (ageGroups : AgeGroups) :
  totalPopulation = 300 →
  totalSampleSize = 30 →
  ageGroups.over40 = 50 →
  ageGroups.between30and40 = 150 →
  ageGroups.under30 = 100 →
  let sample := calculateStratifiedSample totalPopulation totalSampleSize ageGroups
  sample.over40 = 5 ∧ sample.between30and40 = 15 ∧ sample.under30 = 10 :=
by sorry


end stratified_sample_theorem_l1920_192037


namespace classroom_attendance_l1920_192061

theorem classroom_attendance (students_in_restroom : ℕ) 
  (total_students : ℕ) (rows : ℕ) (desks_per_row : ℕ) 
  (occupancy_rate : ℚ) :
  students_in_restroom = 2 →
  total_students = 23 →
  rows = 4 →
  desks_per_row = 6 →
  occupancy_rate = 2/3 →
  ∃ (m : ℕ), m * students_in_restroom - 1 = 
    total_students - (↑(rows * desks_per_row) * occupancy_rate).floor - students_in_restroom ∧
  m = 4 := by
sorry

end classroom_attendance_l1920_192061


namespace pushup_difference_l1920_192002

theorem pushup_difference (david_pushups : ℕ) (total_pushups : ℕ) (zachary_pushups : ℕ) :
  david_pushups = 51 →
  total_pushups = 53 →
  david_pushups > zachary_pushups →
  total_pushups = david_pushups + zachary_pushups →
  david_pushups - zachary_pushups = 49 := by
  sorry

end pushup_difference_l1920_192002


namespace intersection_points_form_hyperbola_l1920_192086

/-- The points of intersection of the given lines form a hyperbola -/
theorem intersection_points_form_hyperbola :
  ∀ (s x y : ℝ), 
    (2*s*x - 3*y - 5*s = 0) → 
    (2*x - 3*s*y + 4 = 0) → 
    ∃ (a b : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 :=
by sorry

end intersection_points_form_hyperbola_l1920_192086


namespace magician_decks_l1920_192082

-- Define the problem parameters
def price_per_deck : ℕ := 2
def total_earnings : ℕ := 4
def decks_left : ℕ := 3

-- Define the theorem
theorem magician_decks : 
  ∃ (initial_decks : ℕ), 
    initial_decks * price_per_deck - total_earnings = decks_left * price_per_deck ∧ 
    initial_decks = 5 := by
  sorry

end magician_decks_l1920_192082


namespace apple_box_weight_l1920_192008

/-- Given a box of apples with total weight and weight after removing half the apples,
    prove the weight of the box and the weight of the apples. -/
theorem apple_box_weight (total_weight : ℝ) (half_removed_weight : ℝ)
    (h1 : total_weight = 62.8)
    (h2 : half_removed_weight = 31.8) :
    ∃ (box_weight apple_weight : ℝ),
      box_weight = 0.8 ∧
      apple_weight = 62 ∧
      total_weight = box_weight + apple_weight ∧
      half_removed_weight = box_weight + apple_weight / 2 := by
  sorry

end apple_box_weight_l1920_192008


namespace fixed_point_parabola_l1920_192067

theorem fixed_point_parabola :
  ∃ (a b : ℝ), ∀ (k : ℝ), 9 * a^2 + k * a - 5 * k + 3 = b ∧ a = 5 ∧ b = 228 :=
sorry

end fixed_point_parabola_l1920_192067


namespace solution_set_reciprocal_inequality_l1920_192089

theorem solution_set_reciprocal_inequality (x : ℝ) :
  1 / x ≤ 1 ↔ x ∈ Set.Iic 0 ∪ Set.Ici 1 :=
sorry

end solution_set_reciprocal_inequality_l1920_192089


namespace library_book_distributions_l1920_192066

-- Define the total number of books
def total_books : ℕ := 8

-- Define the function that calculates the number of valid distributions
def valid_distributions (n : ℕ) : ℕ :=
  -- Count distributions where books in library range from 1 to (n - 2)
  (Finset.range (n - 2)).card

-- Theorem statement
theorem library_book_distributions : 
  valid_distributions total_books = 6 :=
by
  -- Proof goes here
  sorry

end library_book_distributions_l1920_192066


namespace complex_absolute_value_l1920_192001

theorem complex_absolute_value (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 2 - I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → Complex.abs z₁ = Real.sqrt 5 := by
sorry

end complex_absolute_value_l1920_192001


namespace special_triangle_area_l1920_192000

-- Define a right triangle with a 30° angle and hypotenuse of 20 inches
def special_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧  -- Pythagorean theorem for right triangle
  c = 20 ∧  -- Hypotenuse length
  a / c = 1 / 2  -- Sine of 30° angle (opposite / hypotenuse)

-- Theorem statement
theorem special_triangle_area (a b c : ℝ) 
  (h : special_triangle a b c) : a * b / 2 = 50 * Real.sqrt 3 := by
  sorry

end special_triangle_area_l1920_192000


namespace comparison_of_powers_l1920_192030

theorem comparison_of_powers (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a ≠ b) : 
  (a^a * b^b > a^b * b^a) ∧ 
  (a^a * b^b * c^c > (a*b*c)^((a+b+c)/3)) := by
  sorry

end comparison_of_powers_l1920_192030


namespace joshua_needs_32_cents_l1920_192028

/-- The amount of additional cents Joshua needs to purchase a pen -/
def additional_cents_needed (pen_cost : ℕ) (joshua_money : ℕ) (borrowed_amount : ℕ) : ℕ :=
  pen_cost - (joshua_money + borrowed_amount)

/-- Theorem: Joshua needs 32 more cents to buy the pen -/
theorem joshua_needs_32_cents :
  additional_cents_needed 600 500 68 = 32 := by
  sorry

end joshua_needs_32_cents_l1920_192028


namespace smallest_k_for_same_color_square_l1920_192055

/-- 
Given a positive integer n, this theorem states that 2n^2 - n + 1 is the smallest positive 
integer k such that any coloring of a 2n × k table with n colors contains two rows and 
two columns intersecting in four squares of the same color.
-/
theorem smallest_k_for_same_color_square (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k = 2*n^2 - n + 1 ∧ 
  (∀ (table : Fin (2*n) → Fin k → Fin n), 
    ∃ (r1 r2 : Fin (2*n)) (c1 c2 : Fin k),
      r1 ≠ r2 ∧ c1 ≠ c2 ∧
      table r1 c1 = table r1 c2 ∧
      table r1 c1 = table r2 c1 ∧
      table r1 c1 = table r2 c2) ∧
  (∀ k' : ℕ, k' < k → 
    ∃ (table : Fin (2*n) → Fin k' → Fin n), 
      ∀ (r1 r2 : Fin (2*n)) (c1 c2 : Fin k'),
        r1 = r2 ∨ c1 = c2 ∨
        table r1 c1 ≠ table r1 c2 ∨
        table r1 c1 ≠ table r2 c1 ∨
        table r1 c1 ≠ table r2 c2) :=
by sorry

end smallest_k_for_same_color_square_l1920_192055


namespace sum_of_numbers_ge_1_1_l1920_192080

theorem sum_of_numbers_ge_1_1 : 
  let numbers : List ℝ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let filtered_numbers := numbers.filter (λ x => x ≥ 1.1)
  filtered_numbers.sum = 3.9 := by sorry

end sum_of_numbers_ge_1_1_l1920_192080


namespace sequence_properties_l1920_192045

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

def T (n : ℕ) : ℝ := sorry

theorem sequence_properties (n : ℕ) :
  n > 0 →
  (S n = 2 * sequence_a n - 2) →
  (sequence_a n = 2^n ∧ T n = 2^(n+2) - 4 - 2*n) :=
by sorry

end sequence_properties_l1920_192045


namespace vector_subtraction_l1920_192081

/-- Given two vectors AB and AC in 2D space, prove that BC is their difference. -/
theorem vector_subtraction (AB AC : Fin 2 → ℝ) (h1 : AB = ![3, 6]) (h2 : AC = ![1, 2]) :
  AB - AC = ![-2, -4] := by sorry

end vector_subtraction_l1920_192081


namespace sales_revenue_error_l1920_192050

theorem sales_revenue_error (x z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →
  (10 ≤ z ∧ z ≤ 99) →
  (1000 * z + 10 * x) - (1000 * x + 10 * z) = 2920 →
  z = x + 3 ∧ 10 ≤ x ∧ x ≤ 96 :=
by sorry

end sales_revenue_error_l1920_192050


namespace simplify_fraction_product_l1920_192043

theorem simplify_fraction_product : (125 : ℚ) / 5000 * 40 = 1 := by
  sorry

end simplify_fraction_product_l1920_192043


namespace box_tie_length_l1920_192017

/-- Calculates the length of string used to tie a box given the initial length,
    the amount given away, and the fraction of the remainder used. -/
def string_used_for_box (initial_length : ℝ) (given_away : ℝ) (fraction_used : ℚ) : ℝ :=
  (initial_length - given_away) * (fraction_used : ℝ)

/-- Proves that given a string of 90 cm, after removing 30 cm, and using 8/15 of the remainder,
    the length used to tie the box is 32 cm. -/
theorem box_tie_length : 
  string_used_for_box 90 30 (8/15) = 32 := by sorry

end box_tie_length_l1920_192017


namespace polynomial_division_remainder_l1920_192039

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ), 
  (X^4 : Polynomial ℝ) + 3 * X^2 - 2 = (X^2 - 4 * X + 3) * q + r ∧ 
  r = 88 * X - 59 ∧ 
  r.degree < (X^2 - 4 * X + 3).degree := by sorry

end polynomial_division_remainder_l1920_192039


namespace min_paper_length_l1920_192065

/-- Represents a binary message of length 2016 -/
def Message := Fin 2016 → Bool

/-- Represents a paper of length n with 10 pre-colored consecutive squares -/
structure Paper (n : ℕ) where
  squares : Fin n → Bool
  precolored_start : Fin (n - 9)
  precolored : Fin 10 → Bool

/-- A strategy for encoding and decoding messages -/
structure Strategy (n : ℕ) where
  encode : Message → Paper n → Paper n
  decode : Paper n → Message

/-- The strategy works with perfect accuracy -/
def perfect_accuracy (s : Strategy n) : Prop :=
  ∀ (m : Message) (p : Paper n), s.decode (s.encode m p) = m

/-- The main theorem: The minimum value of n for which a perfect strategy exists is 2026 -/
theorem min_paper_length :
  (∃ (s : Strategy 2026), perfect_accuracy s) ∧
  (∀ (n : ℕ), n < 2026 → ¬∃ (s : Strategy n), perfect_accuracy s) :=
sorry

end min_paper_length_l1920_192065


namespace num_multicolor_ducks_l1920_192059

/-- The number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- The number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish per multicolored duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def num_white_ducks : ℕ := 3

/-- The number of black ducks -/
def num_black_ducks : ℕ := 7

/-- The total number of fish in the lake -/
def total_fish : ℕ := 157

/-- The theorem stating the number of multicolored ducks -/
theorem num_multicolor_ducks : ℕ := by
  sorry

#check num_multicolor_ducks

end num_multicolor_ducks_l1920_192059


namespace larger_number_proof_l1920_192040

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) →
  (∃ (k : ℕ+), Nat.lcm a b = 23 * 15 * 16 * k) →
  (max a b = 368) :=
by sorry

end larger_number_proof_l1920_192040


namespace sum_of_polynomials_l1920_192033

-- Define the polynomials
def f (x : ℝ) : ℝ := -6 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -7 * x^2 + 6 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 3

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -6 * x^3 - 5 * x^2 + 15 * x - 11 := by
  sorry

end sum_of_polynomials_l1920_192033


namespace square_garden_perimeter_l1920_192048

/-- The perimeter of a square garden with area 900 square meters is 120 meters and 12000 centimeters. -/
theorem square_garden_perimeter :
  ∀ (side : ℝ), 
  side^2 = 900 →
  (4 * side = 120) ∧ (4 * side * 100 = 12000) := by
  sorry

end square_garden_perimeter_l1920_192048


namespace omelette_combinations_l1920_192034

/-- The number of available fillings for omelettes -/
def num_fillings : ℕ := 8

/-- The number of egg choices for the omelette base -/
def num_egg_choices : ℕ := 4

/-- The total number of omelette combinations -/
def total_combinations : ℕ := 2^num_fillings * num_egg_choices

theorem omelette_combinations : total_combinations = 1024 := by
  sorry

end omelette_combinations_l1920_192034


namespace calculate_brads_speed_l1920_192078

/-- Given two people walking towards each other, calculate the speed of one person given the other's speed and distance traveled. -/
theorem calculate_brads_speed (maxwell_speed brad_speed : ℝ) (total_distance maxwell_distance : ℝ) : 
  maxwell_speed = 2 →
  total_distance = 36 →
  maxwell_distance = 12 →
  2 * maxwell_distance = total_distance →
  brad_speed = 4 := by sorry

end calculate_brads_speed_l1920_192078


namespace store_pricing_l1920_192046

theorem store_pricing (shirts_total : ℝ) (sweaters_total : ℝ) (jeans_total : ℝ)
  (shirts_count : ℕ) (sweaters_count : ℕ) (jeans_count : ℕ)
  (shirt_discount : ℝ) (sweater_discount : ℝ) (jeans_discount : ℝ)
  (h1 : shirts_total = 360)
  (h2 : sweaters_total = 900)
  (h3 : jeans_total = 1200)
  (h4 : shirts_count = 20)
  (h5 : sweaters_count = 45)
  (h6 : jeans_count = 30)
  (h7 : shirt_discount = 2)
  (h8 : sweater_discount = 4)
  (h9 : jeans_discount = 3) :
  let shirt_avg := (shirts_total / shirts_count) - shirt_discount
  let sweater_avg := (sweaters_total / sweaters_count) - sweater_discount
  let jeans_avg := (jeans_total / jeans_count) - jeans_discount
  shirt_avg = sweater_avg ∧ jeans_avg - sweater_avg = 21 := by
  sorry

end store_pricing_l1920_192046


namespace sqrt_expression_equals_three_halves_l1920_192038

theorem sqrt_expression_equals_three_halves :
  (Real.sqrt 48 + (1/4) * Real.sqrt 12) / Real.sqrt 27 = 3/2 := by
  sorry

end sqrt_expression_equals_three_halves_l1920_192038


namespace square_area_with_two_edge_representations_l1920_192011

theorem square_area_with_two_edge_representations (x : ℝ) :
  (3 * x - 12 = 18 - 2 * x) →
  (3 * x - 12)^2 = 36 := by
  sorry

end square_area_with_two_edge_representations_l1920_192011


namespace polynomial_arrangement_l1920_192064

-- Define the polynomial as a function
def polynomial (x y : ℝ) : ℝ := 2 * x^3 * y - 4 * y^2 + 5 * x^2

-- Define the arranged polynomial as a function
def arranged_polynomial (x y : ℝ) : ℝ := 5 * x^2 + 2 * x^3 * y - 4 * y^2

-- Theorem stating that the arranged polynomial is equal to the original polynomial
theorem polynomial_arrangement (x y : ℝ) : 
  polynomial x y = arranged_polynomial x y := by
  sorry

end polynomial_arrangement_l1920_192064


namespace inductive_reasoning_is_specific_to_general_l1920_192041

-- Define the types of reasoning
inductive ReasoningType
  | Analogical
  | Deductive
  | Inductive
  | Emotional

-- Define the direction of reasoning
inductive ReasoningDirection
  | SpecificToGeneral
  | GeneralToSpecific
  | Other

-- Function to get the direction of a reasoning type
def getReasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.SpecificToGeneral
  | _ => ReasoningDirection.Other

-- Theorem statement
theorem inductive_reasoning_is_specific_to_general :
  ∃ (rt : ReasoningType), getReasoningDirection rt = ReasoningDirection.SpecificToGeneral ∧
  rt = ReasoningType.Inductive :=
sorry

end inductive_reasoning_is_specific_to_general_l1920_192041


namespace cos_pi_half_plus_alpha_l1920_192096

-- Define the angle α
def α : Real := sorry

-- Define the point P₀
def P₀ : ℝ × ℝ := (-3, -4)

-- Theorem statement
theorem cos_pi_half_plus_alpha (h : (Real.cos α * (-3) = Real.sin α * (-4))) : 
  Real.cos (π / 2 + α) = 4 / 5 := by sorry

end cos_pi_half_plus_alpha_l1920_192096


namespace blue_area_ratio_l1920_192035

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  /-- Side length of the flag -/
  side : ℝ
  /-- Width of the cross -/
  cross_width : ℝ
  /-- Assumption that the cross (including blue center) is 36% of total area -/
  cross_area_ratio : cross_width * (4 * side - cross_width) / (side * side) = 0.36

/-- Theorem stating that the blue area is 2% of the total flag area -/
theorem blue_area_ratio (flag : SquareFlag) : 
  (flag.cross_width / flag.side) ^ 2 = 0.02 := by
  sorry

end blue_area_ratio_l1920_192035


namespace vovochka_candy_theorem_l1920_192076

/-- Represents the candy distribution problem --/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- The maximum number of candies that can be kept while satisfying the distribution condition --/
def max_kept_candies (cd : CandyDistribution) : ℕ :=
  cd.total_candies - (cd.num_classmates * (cd.min_group_candies / cd.min_group_size))

/-- Theorem stating the maximum number of candies that can be kept in the given scenario --/
theorem vovochka_candy_theorem (cd : CandyDistribution) 
  (h1 : cd.total_candies = 200)
  (h2 : cd.num_classmates = 25)
  (h3 : cd.min_group_size = 16)
  (h4 : cd.min_group_candies = 100) :
  max_kept_candies cd = 37 := by
  sorry

#eval max_kept_candies { total_candies := 200, num_classmates := 25, min_group_size := 16, min_group_candies := 100 }

end vovochka_candy_theorem_l1920_192076


namespace inequality_proof_l1920_192019

theorem inequality_proof (x y z : ℝ) 
  (h1 : y ≥ 2 * z) 
  (h2 : 2 * z ≥ 4 * x)
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) ≥ 
        16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z) :
  4 * x + y ≥ 4 * z := by
sorry

end inequality_proof_l1920_192019


namespace sphere_surface_area_l1920_192047

theorem sphere_surface_area (r : ℝ) (h : r = 4) : 
  4 * π * r^2 = 64 * π := by
  sorry

end sphere_surface_area_l1920_192047


namespace roses_picked_later_l1920_192094

/-- Calculates the number of roses picked later by a florist -/
theorem roses_picked_later (initial : ℕ) (sold : ℕ) (final : ℕ) : 
  initial ≥ sold → final > initial - sold → final - (initial - sold) = 21 := by
  sorry

end roses_picked_later_l1920_192094


namespace total_cost_is_1400_l1920_192013

def cost_of_suits (off_the_rack_cost : ℕ) (tailoring_cost : ℕ) : ℕ :=
  off_the_rack_cost + (3 * off_the_rack_cost + tailoring_cost)

theorem total_cost_is_1400 :
  cost_of_suits 300 200 = 1400 := by
  sorry

end total_cost_is_1400_l1920_192013


namespace peach_boxes_count_l1920_192014

def peaches_per_basket : ℕ := 23
def num_baskets : ℕ := 7
def peaches_eaten : ℕ := 7
def peaches_per_box : ℕ := 13

theorem peach_boxes_count :
  let total_peaches := peaches_per_basket * num_baskets
  let remaining_peaches := total_peaches - peaches_eaten
  (remaining_peaches / peaches_per_box : ℕ) = 11 := by
  sorry

end peach_boxes_count_l1920_192014


namespace hemisphere_surface_area_l1920_192090

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 3) : 
  3 * π * r^2 = 2 * π * r^2 + π * r^2 := by
sorry

end hemisphere_surface_area_l1920_192090


namespace point_N_coordinates_l1920_192053

def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)

theorem point_N_coordinates : 
  ∀ N : ℝ × ℝ, 
  (N.1 - M.1, N.2 - M.2) = (-3 * a.1, -3 * a.2) → 
  N = (2, 0) := by sorry

end point_N_coordinates_l1920_192053


namespace ring_arrangements_count_l1920_192071

/-- The number of ways to arrange 6 rings out of 10 distinguishable rings on 4 fingers. -/
def ring_arrangements : ℕ :=
  (Nat.choose 10 6) * (Nat.factorial 6) * (Nat.choose 9 3)

/-- The correct number of arrangements is 12672000. -/
theorem ring_arrangements_count : ring_arrangements = 12672000 := by
  sorry

end ring_arrangements_count_l1920_192071


namespace triangle_side_constraint_l1920_192060

theorem triangle_side_constraint (a : ℝ) : 
  (6 > 0 ∧ 1 - 3*a > 0 ∧ 10 > 0) ∧  -- positive side lengths
  (6 + (1 - 3*a) > 10 ∧ 6 + 10 > 1 - 3*a ∧ 10 + (1 - 3*a) > 6) →  -- triangle inequality
  -5 < a ∧ a < -1 :=
by sorry


end triangle_side_constraint_l1920_192060


namespace factor_calculation_l1920_192029

theorem factor_calculation (n f : ℝ) : n = 121 ∧ n * f - 140 = 102 → f = 2 := by
  sorry

end factor_calculation_l1920_192029


namespace max_value_fraction_l1920_192042

theorem max_value_fraction (x : ℝ) : 
  (3 * x^2 + 9 * x + 20) / (3 * x^2 + 9 * x + 7) ≤ 53 ∧ 
  ∀ ε > 0, ∃ y : ℝ, (3 * y^2 + 9 * y + 20) / (3 * y^2 + 9 * y + 7) > 53 - ε :=
by sorry

end max_value_fraction_l1920_192042


namespace investment_problem_l1920_192077

/-- Proves that given the conditions of the investment problem, the invested sum is 4200 --/
theorem investment_problem (P : ℝ) 
  (h1 : P * (15 / 100) * 2 - P * (10 / 100) * 2 = 840) : 
  P = 4200 := by
  sorry

end investment_problem_l1920_192077


namespace odd_function_condition_l1920_192049

def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -(f a b x)) ↔ a = 0 ∧ b = 0 := by sorry

end odd_function_condition_l1920_192049


namespace point_on_765_degree_angle_l1920_192083

/-- Given that a point (4, m) lies on the terminal side of an angle of 765°, prove that m = 4 -/
theorem point_on_765_degree_angle (m : ℝ) : 
  (∃ (θ : ℝ), θ = 765 * Real.pi / 180 ∧ Real.tan θ = m / 4) → m = 4 := by
  sorry

end point_on_765_degree_angle_l1920_192083


namespace probability_of_selecting_particular_student_l1920_192051

/-- The probability of selecting a particular student from an institute with multiple classes. -/
theorem probability_of_selecting_particular_student
  (total_classes : ℕ)
  (students_per_class : ℕ)
  (selected_students : ℕ)
  (h1 : total_classes = 8)
  (h2 : students_per_class = 40)
  (h3 : selected_students = 3)
  (h4 : selected_students ≤ total_classes) :
  (selected_students : ℚ) / (total_classes * students_per_class : ℚ) = 3 / 320 := by
  sorry

#check probability_of_selecting_particular_student

end probability_of_selecting_particular_student_l1920_192051


namespace equation_solution_l1920_192021

theorem equation_solution : 
  ∃! x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ∧ x = -15 := by
  sorry

end equation_solution_l1920_192021


namespace roger_trips_l1920_192018

def trays_per_trip : ℕ := 4
def total_trays : ℕ := 12

theorem roger_trips : (total_trays + trays_per_trip - 1) / trays_per_trip = 3 := by
  sorry

end roger_trips_l1920_192018


namespace equal_derivative_points_l1920_192068

theorem equal_derivative_points (x₀ : ℝ) : 
  (2 * x₀ = -3 * x₀^2) → (x₀ = 0 ∨ x₀ = -2/3) :=
by sorry

end equal_derivative_points_l1920_192068


namespace smallest_shift_for_even_function_l1920_192032

theorem smallest_shift_for_even_function (f g : ℝ → ℝ) (σ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x + π / 3)) →
  (∀ x, g x = f (x + σ)) →
  (∀ x, g (-x) = g x) →
  σ > 0 →
  (∀ σ' > 0, (∀ x, f (x + σ') = f (-x + σ')) → σ' ≥ σ) →
  σ = π / 12 := by sorry

end smallest_shift_for_even_function_l1920_192032


namespace ratio_problem_l1920_192052

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : (a + b) / (b + c) = 4/15 := by
  sorry

end ratio_problem_l1920_192052


namespace derivative_f_at_2_l1920_192095

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

-- State the theorem
theorem derivative_f_at_2 : 
  deriv f 2 = 4 := by sorry

end derivative_f_at_2_l1920_192095


namespace club_membership_l1920_192025

theorem club_membership (total_members event_participants : ℕ) 
  (h_total : total_members = 30)
  (h_event : event_participants = 18)
  (h_participation : ∃ (men women : ℕ), 
    men + women = total_members ∧ 
    men + (women / 3) = event_participants) : 
  ∃ (men : ℕ), men = 12 ∧ 
    ∃ (women : ℕ), men + women = total_members ∧ 
      men + (women / 3) = event_participants :=
sorry

end club_membership_l1920_192025


namespace tangent_sum_product_l1920_192007

theorem tangent_sum_product (α β : ℝ) : 
  let γ := Real.arctan (-Real.tan (α + β))
  Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ := by
  sorry

end tangent_sum_product_l1920_192007


namespace original_fraction_l1920_192075

theorem original_fraction (N D : ℚ) : 
  (N * (1 + 30/100)) / (D * (1 - 15/100)) = 25/21 →
  N / D = 425/546 := by
  sorry

end original_fraction_l1920_192075


namespace max_value_sum_of_reciprocals_l1920_192010

theorem max_value_sum_of_reciprocals (a b : ℝ) (h : a + b = 2) :
  (∃ (x : ℝ), ∀ (y : ℝ), (1 / (a^2 + 1) + 1 / (b^2 + 1)) ≤ y) ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (a' b' : ℝ), a' + b' = 2 ∧ 
    (1 / (a'^2 + 1) + 1 / (b'^2 + 1)) > (Real.sqrt 2 + 1) / 2 - ε) :=
by sorry

end max_value_sum_of_reciprocals_l1920_192010


namespace discount_is_ten_percent_l1920_192004

/-- Calculates the discount percentage on a retail price given wholesale price, retail price, and profit percentage. -/
def discount_percentage (wholesale_price retail_price profit_percentage : ℚ) : ℚ :=
  let profit := wholesale_price * profit_percentage
  let selling_price := wholesale_price + profit
  let discount_amount := retail_price - selling_price
  (discount_amount / retail_price) * 100

/-- Theorem stating that the discount percentage is 10% given the problem conditions. -/
theorem discount_is_ten_percent :
  discount_percentage 108 144 0.2 = 10 := by
  sorry

end discount_is_ten_percent_l1920_192004


namespace inverse_proportion_problem_l1920_192022

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ x0 y0, x0 + y0 = 36 ∧ x0 = 3 * y0 ∧ x0 * y0 = k) :
  x = -6 → y = -40.5 := by
  sorry

end inverse_proportion_problem_l1920_192022


namespace batsman_average_increase_l1920_192054

theorem batsman_average_increase 
  (innings : Nat) 
  (last_score : Nat) 
  (final_average : Nat) 
  (h1 : innings = 12) 
  (h2 : last_score = 75) 
  (h3 : final_average = 64) : 
  (final_average : ℚ) - (((innings : ℚ) * final_average - last_score) / (innings - 1)) = 1 := by
  sorry

end batsman_average_increase_l1920_192054


namespace jinas_mascots_l1920_192044

/-- The number of mascots Jina has -/
def total_mascots (x y z : ℕ) : ℕ := x + y + z

/-- The theorem stating the total number of Jina's mascots -/
theorem jinas_mascots :
  ∃ (x y z : ℕ),
    y = 3 * x ∧
    z = 2 * y ∧
    (x + (5/2 : ℚ) * y) / y = 3/7 ∧
    total_mascots x y z = 60 := by
  sorry


end jinas_mascots_l1920_192044


namespace ascending_order_of_powers_of_two_l1920_192026

theorem ascending_order_of_powers_of_two :
  let a := (2 : ℝ) ^ (1/3 : ℝ)
  let b := (2 : ℝ) ^ (3/8 : ℝ)
  let c := (2 : ℝ) ^ (2/5 : ℝ)
  let d := (2 : ℝ) ^ (4/9 : ℝ)
  let e := (2 : ℝ) ^ (1/2 : ℝ)
  a < b ∧ b < c ∧ c < d ∧ d < e := by sorry

end ascending_order_of_powers_of_two_l1920_192026


namespace weight_system_properties_l1920_192074

/-- Represents a set of weights -/
def Weights : List ℕ := [1, 3, 9, 27]

/-- The maximum weight that can be measured -/
def MaxWeight : ℕ := 40

/-- Checks if a weight can be represented by a combination of given weights -/
def isRepresentable (n : ℕ) (weights : List ℕ) : Prop :=
  ∃ (combination : List Bool), 
    combination.length = weights.length ∧ 
    (List.zip combination weights).foldl (λ sum (b, w) => sum + if b then w else 0) 0 = n

theorem weight_system_properties :
  (∀ n : ℕ, n ≤ MaxWeight → isRepresentable n Weights) ∧
  (∀ n : ℕ, n > MaxWeight → ¬ isRepresentable n Weights) :=
sorry

end weight_system_properties_l1920_192074


namespace arithmetic_sequence_sum_l1920_192079

/-- The sum of terms in an arithmetic sequence with first term 2, common difference 12, and last term 182 is 1472 -/
theorem arithmetic_sequence_sum : 
  let a₁ : ℕ := 2  -- First term
  let d : ℕ := 12  -- Common difference
  let aₙ : ℕ := 182  -- Last term
  let n : ℕ := (aₙ - a₁) / d + 1  -- Number of terms
  (n : ℝ) * (a₁ + aₙ) / 2 = 1472 := by sorry

end arithmetic_sequence_sum_l1920_192079


namespace fractional_equation_solution_l1920_192097

theorem fractional_equation_solution (x : ℝ) :
  x ≠ 2 → x ≠ 0 → (1 / (x - 2) = 3 / x) ↔ x = 3 :=
by sorry

end fractional_equation_solution_l1920_192097


namespace intersection_of_A_and_B_l1920_192091

def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l1920_192091


namespace multiply_333_by_111_l1920_192093

theorem multiply_333_by_111 : 333 * 111 = 36963 := by
  sorry

end multiply_333_by_111_l1920_192093


namespace largest_valid_number_l1920_192024

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit integer
  (n / 100 = 8) ∧  -- starts with 8
  (∀ d, d ≠ 0 ∧ d ∈ [n / 100, (n / 10) % 10, n % 10] → n % d = 0)  -- divisible by each distinct, non-zero digit

theorem largest_valid_number :
  is_valid_number 864 ∧ ∀ n, is_valid_number n → n ≤ 864 :=
sorry

end largest_valid_number_l1920_192024


namespace cricket_match_analysis_l1920_192062

-- Define the cricket match parameters
def total_overs : ℕ := 50
def initial_overs : ℕ := 10
def remaining_overs : ℕ := total_overs - initial_overs
def initial_run_rate : ℚ := 32/10
def initial_wickets : ℕ := 2
def target_score : ℕ := 320
def min_additional_wickets : ℕ := 5

-- Define the theorem
theorem cricket_match_analysis :
  let initial_score := initial_run_rate * initial_overs
  let remaining_score := target_score - initial_score
  let required_run_rate := remaining_score / remaining_overs
  let total_wickets_needed := initial_wickets + min_additional_wickets
  (required_run_rate = 72/10) ∧ (total_wickets_needed = 7) := by
  sorry

end cricket_match_analysis_l1920_192062


namespace food_combo_discount_percentage_l1920_192073

/-- Calculates the discount percentage on food combos during a special offer. -/
theorem food_combo_discount_percentage
  (evening_ticket_cost : ℚ)
  (food_combo_cost : ℚ)
  (ticket_discount_percent : ℚ)
  (total_savings : ℚ)
  (h1 : evening_ticket_cost = 10)
  (h2 : food_combo_cost = 10)
  (h3 : ticket_discount_percent = 20)
  (h4 : total_savings = 7) :
  (total_savings - ticket_discount_percent / 100 * evening_ticket_cost) / food_combo_cost * 100 = 50 := by
sorry

end food_combo_discount_percentage_l1920_192073


namespace range_of_a_for_false_proposition_l1920_192006

theorem range_of_a_for_false_proposition :
  (∀ x ∈ Set.Icc 0 1, 2 * x + a ≥ 0) ↔ a > 0 :=
by sorry

end range_of_a_for_false_proposition_l1920_192006


namespace hyperbola_standard_equation_l1920_192005

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a, b > 0,
    focal length 10, and point P(3, 4) on one of its asymptotes,
    prove that the standard equation of C is x²/9 - y²/16 = 1 -/
theorem hyperbola_standard_equation 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hf : (2 : ℝ) * Real.sqrt (a^2 + b^2) = 10) 
  (hp : (3 : ℝ)^2 / a^2 - (4 : ℝ)^2 / b^2 = 0) :
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 9 - y^2 / 16 = 1 := by
  sorry

end hyperbola_standard_equation_l1920_192005


namespace paco_sweet_cookies_left_l1920_192036

/-- The number of sweet cookies Paco has left -/
def sweet_cookies_left (initial_sweet : ℕ) (eaten_sweet : ℕ) : ℕ :=
  initial_sweet - eaten_sweet

/-- Theorem: Paco has 19 sweet cookies left -/
theorem paco_sweet_cookies_left : 
  sweet_cookies_left 34 15 = 19 := by
  sorry

end paco_sweet_cookies_left_l1920_192036


namespace two_numbers_between_4_and_16_l1920_192087

theorem two_numbers_between_4_and_16 :
  ∃ (a b : ℝ), 
    4 < a ∧ a < b ∧ b < 16 ∧
    (b - a = a - 4) ∧
    (b * b = a * 16) ∧
    a + b = 20 := by
sorry

end two_numbers_between_4_and_16_l1920_192087


namespace terminal_side_in_quadrant_II_l1920_192056

def α : Real := 2

theorem terminal_side_in_quadrant_II :
  π / 2 < α ∧ α < π :=
sorry

end terminal_side_in_quadrant_II_l1920_192056


namespace min_value_geometric_sequence_l1920_192069

def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => geometric_sequence a₁ r n * r

def expression (a₁ r : ℝ) : ℝ :=
  5 * (geometric_sequence a₁ r 1) + 6 * (geometric_sequence a₁ r 2)

theorem min_value_geometric_sequence :
  ∃ (min_val : ℝ), min_val = -25/12 ∧
  ∀ (r : ℝ), expression 2 r ≥ min_val :=
sorry

end min_value_geometric_sequence_l1920_192069


namespace frog_escape_probability_l1920_192072

/-- Probability of the frog surviving when starting at pad N -/
noncomputable def P (N : ℕ) : ℝ :=
  sorry

/-- The number of lilypads -/
def num_pads : ℕ := 21

/-- The starting position of the frog -/
def start_pos : ℕ := 3

theorem frog_escape_probability :
  (∀ N : ℕ, 0 < N → N < num_pads - 1 →
    P N = (2 * N : ℝ) / 20 * P (N - 1) + (1 - (2 * N : ℝ) / 20) * P (N + 1)) →
  P 0 = 0 →
  P (num_pads - 1) = 1 →
  P start_pos = 4 / 11 := by
  sorry

end frog_escape_probability_l1920_192072


namespace intersection_empty_implies_t_geq_one_l1920_192070

theorem intersection_empty_implies_t_geq_one (t : ℝ) : 
  let M : Set ℝ := {x | x ≤ 1}
  let P : Set ℝ := {x | x > t}
  (M ∩ P = ∅) → t ≥ 1 := by
  sorry

end intersection_empty_implies_t_geq_one_l1920_192070


namespace average_problem_l1920_192023

theorem average_problem (x : ℝ) : 
  let numbers := [54, 55, 57, 58, 59, 62, 62, 63, x]
  (numbers.sum / numbers.length : ℝ) = 60 → x = 70 := by
sorry

end average_problem_l1920_192023


namespace arithmetic_geometric_ratio_l1920_192012

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def geometric_sequence_terms (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : arithmetic_sequence a d) 
  (h3 : geometric_sequence_terms (a 1) (a 3) (a 9)) : 
  3 * (a 3) / (a 16) = 9 / 16 := by
sorry

end arithmetic_geometric_ratio_l1920_192012


namespace coals_per_bag_prove_coals_per_bag_l1920_192099

-- Define the constants from the problem
def coals_per_set : ℕ := 15
def minutes_per_set : ℕ := 20
def total_minutes : ℕ := 240
def num_bags : ℕ := 3

-- Define the theorem
theorem coals_per_bag : ℕ :=
  let sets_burned := total_minutes / minutes_per_set
  let total_coals_burned := sets_burned * coals_per_set
  total_coals_burned / num_bags

-- State the theorem to be proved
theorem prove_coals_per_bag : coals_per_bag = 60 := by
  sorry

end coals_per_bag_prove_coals_per_bag_l1920_192099


namespace most_frequent_is_mode_l1920_192015

/-- The mode of a dataset is the value that appears most frequently. -/
def mode (dataset : Multiset α) [DecidableEq α] : Set α :=
  {x | ∀ y, dataset.count x ≥ dataset.count y}

/-- The most frequent data in a dataset is the mode. -/
theorem most_frequent_is_mode (dataset : Multiset α) [DecidableEq α] :
  ∀ x ∈ mode dataset, ∀ y, dataset.count x ≥ dataset.count y :=
sorry

end most_frequent_is_mode_l1920_192015


namespace smallest_whole_number_above_sum_sum_less_than_16_sixteen_is_smallest_l1920_192084

theorem smallest_whole_number_above_sum : ℕ → Prop :=
  fun n => (2 + 1/2 + 3 + 1/3 + 4 + 1/4 + 5 + 1/5 : ℚ) < n ∧
           ∀ m : ℕ, (2 + 1/2 + 3 + 1/3 + 4 + 1/4 + 5 + 1/5 : ℚ) < m → n ≤ m

theorem sum_less_than_16 :
  (2 + 1/2 + 3 + 1/3 + 4 + 1/4 + 5 + 1/5 : ℚ) < 16 :=
sorry

theorem sixteen_is_smallest : smallest_whole_number_above_sum 16 :=
sorry

end smallest_whole_number_above_sum_sum_less_than_16_sixteen_is_smallest_l1920_192084


namespace first_year_exceeding_target_l1920_192098

-- Define the initial investment and growth rate
def initial_investment : ℝ := 1.3
def growth_rate : ℝ := 0.12

-- Define the target investment
def target_investment : ℝ := 2.0

-- Define the function to calculate the investment for a given year
def investment (year : ℕ) : ℝ := initial_investment * (1 + growth_rate) ^ (year - 2015)

-- Theorem statement
theorem first_year_exceeding_target : 
  (∀ y : ℕ, y < 2019 → investment y ≤ target_investment) ∧ 
  investment 2019 > target_investment :=
sorry

end first_year_exceeding_target_l1920_192098


namespace area_ratio_quadrilateral_to_dodecagon_l1920_192009

/-- Regular dodecagon with vertices ABCDEFGHIJKL -/
structure RegularDodecagon where
  vertices : Fin 12 → ℝ × ℝ
  is_regular : sorry

/-- Area of a regular dodecagon -/
def area_dodecagon (d : RegularDodecagon) : ℝ := sorry

/-- Area of quadrilateral ACEG in a regular dodecagon -/
def area_quadrilateral_ACEG (d : RegularDodecagon) : ℝ := sorry

/-- Theorem: The ratio of the area of quadrilateral ACEG to the area of a regular dodecagon is 1/(3√3) -/
theorem area_ratio_quadrilateral_to_dodecagon (d : RegularDodecagon) :
  area_quadrilateral_ACEG d / area_dodecagon d = 1 / (3 * Real.sqrt 3) := by sorry

end area_ratio_quadrilateral_to_dodecagon_l1920_192009


namespace geometric_sequence_property_l1920_192016

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 + a 7 = 2 * Real.pi →
  a 6 * (a 4 + 2 * a 6 + a 8) = 4 * Real.pi^2 := by
  sorry

end geometric_sequence_property_l1920_192016


namespace sum_of_squares_greater_than_ten_l1920_192063

theorem sum_of_squares_greater_than_ten (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁₂ : |x₁ - x₂| > 1) (h₁₃ : |x₁ - x₃| > 1) (h₁₄ : |x₁ - x₄| > 1) (h₁₅ : |x₁ - x₅| > 1)
  (h₂₃ : |x₂ - x₃| > 1) (h₂₄ : |x₂ - x₄| > 1) (h₂₅ : |x₂ - x₅| > 1)
  (h₃₄ : |x₃ - x₄| > 1) (h₃₅ : |x₃ - x₅| > 1)
  (h₄₅ : |x₄ - x₅| > 1) :
  x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 > 10 := by
sorry

end sum_of_squares_greater_than_ten_l1920_192063


namespace triangle_similarity_theorem_l1920_192057

/-- Given a triangle ADE with point C on AD and point B on AC, prove that FC = 10 -/
theorem triangle_similarity_theorem 
  (DC : ℝ) (CB : ℝ) (AD : ℝ) (AB : ℝ) (ED : ℝ) (FC : ℝ) :
  DC = 9 →
  CB = 7 →
  AB = (1/3) * AD →
  ED = (2/3) * AD →
  FC = 10 := by
sorry

end triangle_similarity_theorem_l1920_192057
