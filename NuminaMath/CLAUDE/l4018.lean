import Mathlib

namespace parallel_vectors_x_value_l4018_401830

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 1)
  let b : ℝ × ℝ := (x, 2)
  are_parallel a b → x = 8 := by
  sorry

end parallel_vectors_x_value_l4018_401830


namespace volume_of_specific_tetrahedron_l4018_401866

/-- The volume of a tetrahedron with given edge lengths -/
def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ := sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths -/
theorem volume_of_specific_tetrahedron :
  tetrahedron_volume 3 4 6 5 (Real.sqrt 37) (2 * Real.sqrt 10) = (4 * Real.sqrt 77) / 3 := by sorry

end volume_of_specific_tetrahedron_l4018_401866


namespace original_eq_hyperbola_and_ellipse_l4018_401874

-- Define the original equation
def original_equation (x y : ℝ) : Prop := y^4 - 16*x^4 = 8*y^2 - 4

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 - 4*x^2 = 4

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := y^2 + 4*x^2 = 4

-- Theorem stating that the original equation is equivalent to the union of a hyperbola and an ellipse
theorem original_eq_hyperbola_and_ellipse :
  ∀ x y : ℝ, original_equation x y ↔ (hyperbola_equation x y ∨ ellipse_equation x y) :=
sorry

end original_eq_hyperbola_and_ellipse_l4018_401874


namespace arccos_negative_half_l4018_401844

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by
  sorry

end arccos_negative_half_l4018_401844


namespace polynomial_square_l4018_401832

theorem polynomial_square (a b : ℚ) : 
  (∃ q₀ q₁ : ℚ, ∀ x, x^4 + 3*x^3 + x^2 + a*x + b = (x^2 + q₁*x + q₀)^2) → 
  b = 25/64 := by
sorry

end polynomial_square_l4018_401832


namespace younger_person_age_l4018_401846

/-- Given two people's ages, proves that the younger person is 12 years old --/
theorem younger_person_age
  (total_age : ℕ)
  (age_difference : ℕ)
  (h1 : total_age = 30)
  (h2 : age_difference = 6) :
  (total_age - age_difference) / 2 = 12 :=
by sorry

end younger_person_age_l4018_401846


namespace quadratic_real_roots_l4018_401871

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by sorry

end quadratic_real_roots_l4018_401871


namespace smallest_n_for_zero_last_four_digits_l4018_401813

def last_four_digits_zero (n : ℕ) : Prop :=
  ∃ k : ℕ, 225 * 525 * n = k * 10000

theorem smallest_n_for_zero_last_four_digits :
  ∀ n : ℕ, n < 16 → ¬(last_four_digits_zero n) ∧ last_four_digits_zero 16 :=
by sorry

end smallest_n_for_zero_last_four_digits_l4018_401813


namespace apple_eating_time_l4018_401839

theorem apple_eating_time (apples_per_hour : ℕ) (total_apples : ℕ) (h1 : apples_per_hour = 5) (h2 : total_apples = 15) :
  total_apples / apples_per_hour = 3 := by
sorry

end apple_eating_time_l4018_401839


namespace intersection_M_N_l4018_401861

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l4018_401861


namespace bobby_candy_consumption_l4018_401851

theorem bobby_candy_consumption (initial_candy : ℕ) (remaining_candy : ℕ) 
  (h1 : initial_candy = 30) (h2 : remaining_candy = 7) :
  initial_candy - remaining_candy = 23 := by
  sorry

end bobby_candy_consumption_l4018_401851


namespace keith_pears_l4018_401887

theorem keith_pears (mike_pears : ℕ) (keith_gave_away : ℕ) (total_left : ℕ) 
  (h1 : mike_pears = 12)
  (h2 : keith_gave_away = 46)
  (h3 : total_left = 13) :
  ∃ keith_initial : ℕ, 
    keith_initial = 47 ∧ 
    keith_initial - keith_gave_away + mike_pears = total_left :=
by sorry

end keith_pears_l4018_401887


namespace num_small_squares_seven_l4018_401853

/-- The number of small squares formed when a square is divided into n equal parts on each side and the points are joined -/
def num_small_squares (n : ℕ) : ℕ := 4 * (n * (n - 1) / 2)

/-- Theorem stating that the number of small squares is 84 when n = 7 -/
theorem num_small_squares_seven : num_small_squares 7 = 84 := by
  sorry

end num_small_squares_seven_l4018_401853


namespace car_hire_total_amount_l4018_401857

/-- Represents the hire charges for a car -/
structure CarHire where
  hourly_rate : ℕ
  hours_a : ℕ
  hours_b : ℕ
  hours_c : ℕ
  amount_b : ℕ

/-- Calculates the total amount paid for hiring the car -/
def total_amount (hire : CarHire) : ℕ :=
  hire.hourly_rate * (hire.hours_a + hire.hours_b + hire.hours_c)

/-- Theorem stating the total amount paid for the car hire -/
theorem car_hire_total_amount (hire : CarHire)
  (h1 : hire.hours_a = 7)
  (h2 : hire.hours_b = 8)
  (h3 : hire.hours_c = 11)
  (h4 : hire.amount_b = 160)
  (h5 : hire.hourly_rate = hire.amount_b / hire.hours_b) :
  total_amount hire = 520 := by
  sorry

#check car_hire_total_amount

end car_hire_total_amount_l4018_401857


namespace nth_equation_l4018_401884

theorem nth_equation (n : ℕ) (hn : n > 0) :
  1 + 1 / n - 2 / (2 * n + 1) = (2 * n^2 + n + 1) / (n * (2 * n + 1)) := by
  sorry

end nth_equation_l4018_401884


namespace cube_monotone_l4018_401855

theorem cube_monotone (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_monotone_l4018_401855


namespace distance_to_point_distance_from_origin_to_point_l4018_401815

theorem distance_to_point : ℝ → ℝ → ℝ
  | x, y => Real.sqrt (x^2 + y^2)

theorem distance_from_origin_to_point :
  distance_to_point 8 (-15) = 17 := by
  sorry

end distance_to_point_distance_from_origin_to_point_l4018_401815


namespace sortable_configurations_after_three_passes_l4018_401805

/-- The number of sortable book configurations after three passes -/
def sortableConfigurations (n : ℕ) : ℕ :=
  6 * 4^(n - 3)

/-- Theorem stating the number of sortable configurations for n ≥ 3 books after three passes -/
theorem sortable_configurations_after_three_passes (n : ℕ) (h : n ≥ 3) :
  sortableConfigurations n = 6 * 4^(n - 3) := by
  sorry

end sortable_configurations_after_three_passes_l4018_401805


namespace quadratic_equation_roots_l4018_401868

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + 2*m - 4 = 0 ∧ x₂^2 - x₂ + 2*m - 4 = 0) →
  (m ≤ 17/8 ∧
   (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - x₁ + 2*m - 4 = 0 ∧ x₂^2 - x₂ + 2*m - 4 = 0 →
    (x₁ - 3) * (x₂ - 3) = m^2 - 1 → m = -1)) :=
by sorry

end quadratic_equation_roots_l4018_401868


namespace special_function_value_l4018_401817

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  f 0 = 1008 ∧
  (∀ x : ℝ, f (x + 4) - f x ≤ 2 * (x + 1)) ∧
  (∀ x : ℝ, f (x + 12) - f x ≥ 6 * (x + 5))

/-- The main theorem -/
theorem special_function_value (f : ℝ → ℝ) (h : SpecialFunction f) :
  f 2016 / 2016 = 504 := by
  sorry

end special_function_value_l4018_401817


namespace rectangle_area_theorem_l4018_401892

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ := sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- Finds the point E on CD that is one-fourth the way from C to D -/
def findPointE (C D : Point) : Point := sorry

/-- Finds the intersection point F of BE and AC -/
def findIntersectionF (A B C E : Point) : Point := sorry

theorem rectangle_area_theorem (ABCD : Rectangle) :
  let E := findPointE ABCD.C ABCD.D
  let F := findIntersectionF ABCD.A ABCD.B ABCD.C E
  quadrilateralArea ABCD.A F E ABCD.D = 36 →
  rectangleArea ABCD = 144 := by sorry

end rectangle_area_theorem_l4018_401892


namespace candy_distribution_l4018_401883

theorem candy_distribution (e : ℚ) 
  (frank_candies : ℚ) (gail_candies : ℚ) (hank_candies : ℚ) : 
  frank_candies = 4 * e →
  gail_candies = 4 * frank_candies →
  hank_candies = 6 * gail_candies →
  e + frank_candies + gail_candies + hank_candies = 876 →
  e = 7.5 := by
sorry


end candy_distribution_l4018_401883


namespace orange_ribbons_l4018_401837

theorem orange_ribbons (total : ℕ) (yellow purple orange silver : ℕ) : 
  yellow + purple + orange + silver = total →
  4 * yellow = total →
  3 * purple = total →
  6 * orange = total →
  silver = 40 →
  orange = 27 :=
by
  sorry

end orange_ribbons_l4018_401837


namespace infinite_square_divisibility_l4018_401877

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | (n + 4) => 2 * a (n + 3) + a (n + 2) - 2 * a (n + 1) - a n

theorem infinite_square_divisibility :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (n : ℤ)^2 ∣ a n := by sorry

end infinite_square_divisibility_l4018_401877


namespace advertising_time_l4018_401894

def newscast_duration : ℕ := 30
def national_news_duration : ℕ := 12
def international_news_duration : ℕ := 5
def sports_duration : ℕ := 5
def weather_forecast_duration : ℕ := 2

theorem advertising_time :
  newscast_duration - (national_news_duration + international_news_duration + sports_duration + weather_forecast_duration) = 6 := by
  sorry

end advertising_time_l4018_401894


namespace inequality_proof_l4018_401807

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (a*b + 1)/(a + b)^2 + (b*c + 1)/(b + c)^2 + (c*a + 1)/(c + a)^2 ≥ 3 := by
sorry

end inequality_proof_l4018_401807


namespace shorter_leg_length_is_five_l4018_401888

/-- A right triangle that can be cut and reassembled into a square -/
structure CuttableRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  is_right_triangle : shorter_leg^2 + longer_leg^2 = hypotenuse^2
  can_form_square : hypotenuse = 2 * shorter_leg

/-- The theorem stating that if a right triangle with longer leg 10 can be cut and
    reassembled into a square, then its shorter leg has length 5 -/
theorem shorter_leg_length_is_five
  (triangle : CuttableRightTriangle)
  (h : triangle.longer_leg = 10) :
  triangle.shorter_leg = 5 := by
  sorry


end shorter_leg_length_is_five_l4018_401888


namespace diego_orange_weight_l4018_401835

/-- Given Diego's fruit purchases, prove the weight of oranges he bought. -/
theorem diego_orange_weight (total_capacity : ℕ) (watermelon_weight : ℕ) (grape_weight : ℕ) (apple_weight : ℕ) 
  (h1 : total_capacity = 20)
  (h2 : watermelon_weight = 1)
  (h3 : grape_weight = 1)
  (h4 : apple_weight = 17) :
  total_capacity - (watermelon_weight + grape_weight + apple_weight) = 1 := by
  sorry

end diego_orange_weight_l4018_401835


namespace extended_segment_vector_representation_l4018_401881

/-- Given a line segment AB extended past B to Q with AQ:QB = 7:2,
    prove that Q = (2/9)A + (7/9)B -/
theorem extended_segment_vector_representation 
  (A B Q : ℝ × ℝ) -- Points in 2D plane
  (h : (dist A Q) / (dist Q B) = 7 / 2) -- AQ:QB = 7:2
  : ∃ (x y : ℝ), x = 2/9 ∧ y = 7/9 ∧ Q = x • A + y • B :=
by sorry


end extended_segment_vector_representation_l4018_401881


namespace vector_to_line_parallel_to_direction_l4018_401897

/-- A line parameterized by x = 3t + 3, y = 2t + 3 -/
def parametric_line (t : ℝ) : ℝ × ℝ := (3*t + 3, 2*t + 3)

/-- The vector we want to prove is correct -/
def vector : ℝ × ℝ := (9, 6)

/-- The direction vector -/
def direction : ℝ × ℝ := (3, 2)

theorem vector_to_line_parallel_to_direction :
  ∃ (t : ℝ), parametric_line t = vector ∧ 
  ∃ (k : ℝ), vector.1 = k * direction.1 ∧ vector.2 = k * direction.2 := by
sorry

end vector_to_line_parallel_to_direction_l4018_401897


namespace school_sports_probabilities_l4018_401809

/-- Represents a school with boys and girls, some of whom like sports -/
structure School where
  girls : ℕ
  boys : ℕ
  boys_like_sports : ℕ
  girls_like_sports : ℕ
  boys_ratio : boys = 3 * girls / 2
  boys_sports_ratio : boys_like_sports = 2 * boys / 5
  girls_sports_ratio : girls_like_sports = girls / 5

/-- The probability that a randomly selected student likes sports -/
def prob_likes_sports (s : School) : ℚ :=
  (s.boys_like_sports + s.girls_like_sports : ℚ) / (s.boys + s.girls)

/-- The probability that a randomly selected student who likes sports is a boy -/
def prob_boy_given_sports (s : School) : ℚ :=
  (s.boys_like_sports : ℚ) / (s.boys_like_sports + s.girls_like_sports)

theorem school_sports_probabilities (s : School) :
  prob_likes_sports s = 8/25 ∧ prob_boy_given_sports s = 3/4 := by
  sorry


end school_sports_probabilities_l4018_401809


namespace first_day_over_500_day_is_saturday_l4018_401850

def paperclips (k : ℕ) : ℕ := 5 * 3^k

theorem first_day_over_500 :
  ∃ k : ℕ, paperclips k > 500 ∧ ∀ j : ℕ, j < k → paperclips j ≤ 500 :=
by sorry

theorem day_is_saturday : 
  ∃ k : ℕ, paperclips k > 500 ∧ ∀ j : ℕ, j < k → paperclips j ≤ 500 → k = 5 :=
by sorry

end first_day_over_500_day_is_saturday_l4018_401850


namespace condition_sufficient_not_necessary_l4018_401803

theorem condition_sufficient_not_necessary (a b : ℝ) :
  ((1 < b) ∧ (b < a)) → (a - 1 > |b - 1|) ∧
  ¬(∀ a b : ℝ, (a - 1 > |b - 1|) → ((1 < b) ∧ (b < a))) :=
by sorry

end condition_sufficient_not_necessary_l4018_401803


namespace solution_sets_equal_l4018_401820

/-- A strictly increasing bijective function from R to R -/
def StrictlyIncreasingBijection (f : ℝ → ℝ) : Prop :=
  Function.Bijective f ∧ StrictMono f

/-- The solution set of x = f(x) -/
def SolutionSetP (f : ℝ → ℝ) : Set ℝ :=
  {x | x = f x}

/-- The solution set of x = f(f(x)) -/
def SolutionSetQ (f : ℝ → ℝ) : Set ℝ :=
  {x | x = f (f x)}

/-- Theorem: For a strictly increasing bijective function f from R to R,
    the solution set P of x = f(x) is equal to the solution set Q of x = f(f(x)) -/
theorem solution_sets_equal (f : ℝ → ℝ) (h : StrictlyIncreasingBijection f) :
  SolutionSetP f = SolutionSetQ f := by
  sorry

end solution_sets_equal_l4018_401820


namespace intersection_of_A_and_B_l4018_401829

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l4018_401829


namespace frog_jump_distance_l4018_401896

theorem frog_jump_distance (grasshopper_distance : ℕ) (difference : ℕ) (frog_distance : ℕ) :
  grasshopper_distance = 13 →
  difference = 2 →
  grasshopper_distance = frog_distance + difference →
  frog_distance = 11 := by
sorry

end frog_jump_distance_l4018_401896


namespace cello_practice_time_l4018_401852

/-- Given a total practice time of 7.5 hours in a week, with 86 minutes of practice on each of 2 days,
    the remaining practice time on the other days is 278 minutes. -/
theorem cello_practice_time (total_hours : ℝ) (practice_minutes_per_day : ℕ) (practice_days : ℕ) :
  total_hours = 7.5 ∧ practice_minutes_per_day = 86 ∧ practice_days = 2 →
  (total_hours * 60 : ℝ) - (practice_minutes_per_day * practice_days : ℕ) = 278 := by
  sorry

end cello_practice_time_l4018_401852


namespace three_numbers_ratio_l4018_401811

theorem three_numbers_ratio (a b c : ℝ) : 
  (a : ℝ) / 2 = (b : ℝ) / 3 ∧ (b : ℝ) / 3 = (c : ℝ) / 4 →
  a^2 + b^2 + c^2 = 725 →
  (a = 10 ∧ b = 15 ∧ c = 20) ∨ (a = -10 ∧ b = -15 ∧ c = -20) :=
by sorry

end three_numbers_ratio_l4018_401811


namespace cabbage_increase_l4018_401806

/-- Represents a square garden where cabbages are grown -/
structure CabbageGarden where
  side : ℕ  -- Side length of the square garden

/-- The number of cabbages in a garden -/
def num_cabbages (g : CabbageGarden) : ℕ := g.side * g.side

/-- Theorem: If the number of cabbages increased by 199 from last year to this year,
    and the garden remained square-shaped, then the number of cabbages this year is 10,000 -/
theorem cabbage_increase (last_year this_year : CabbageGarden) :
  num_cabbages this_year = num_cabbages last_year + 199 →
  num_cabbages this_year = 10000 := by
  sorry

#check cabbage_increase

end cabbage_increase_l4018_401806


namespace trio_selection_l4018_401865

theorem trio_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 3) :
  Nat.choose n k = 220 := by
  sorry

end trio_selection_l4018_401865


namespace unique_number_property_l4018_401808

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Reverse of a natural number -/
def reverseNum (n : ℕ) : ℕ := sorry

/-- Prime factors of a natural number -/
def primeFactors (n : ℕ) : List ℕ := sorry

/-- Remove zeros from a natural number -/
def removeZeros (n : ℕ) : ℕ := sorry

theorem unique_number_property : ∃! n : ℕ, 
  n > 0 ∧ 
  n = sumOfDigits n * reverseNum (sumOfDigits n) ∧ 
  n = removeZeros ((List.sum (List.map (λ x => x^2) (primeFactors n))) / 2) ∧
  n = 1729 := by
  sorry

end unique_number_property_l4018_401808


namespace apples_left_is_ten_l4018_401885

/-- Represents the number of apples picked by Mike -/
def mike_apples : ℕ := 12

/-- Represents the number of apples eaten by Nancy -/
def nancy_apples : ℕ := 7

/-- Represents the number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- Represents the number of pears picked by Keith -/
def keith_pears : ℕ := 4

/-- Represents the number of apples picked by Christine -/
def christine_apples : ℕ := 10

/-- Represents the number of pears picked by Christine -/
def christine_pears : ℕ := 3

/-- Represents the number of bananas picked by Christine -/
def christine_bananas : ℕ := 5

/-- Represents the number of apples eaten by Greg -/
def greg_apples : ℕ := 9

/-- Represents the number of peaches picked by an unknown person -/
def unknown_peaches : ℕ := 14

/-- Represents the number of plums picked by an unknown person -/
def unknown_plums : ℕ := 7

/-- Represents the ratio of pears picked to apples disappeared -/
def pears_per_apple : ℕ := 3

/-- Theorem stating that the number of apples left is 10 -/
theorem apples_left_is_ten : 
  mike_apples + keith_apples + christine_apples - 
  nancy_apples - greg_apples - 
  ((keith_pears + christine_pears) / pears_per_apple) = 10 := by
  sorry

end apples_left_is_ten_l4018_401885


namespace division_remainder_l4018_401800

theorem division_remainder : ∃ (q : ℕ), 37 = 8 * q + 5 ∧ 5 < 8 := by sorry

end division_remainder_l4018_401800


namespace sin_cos_sum_equals_one_l4018_401821

theorem sin_cos_sum_equals_one : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end sin_cos_sum_equals_one_l4018_401821


namespace other_x_intercept_l4018_401870

/-- A quadratic function with vertex (5, 10) and one x-intercept at (-1, 0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a ≠ 0 → -b / (2 * a) = 5
  vertex_y : a ≠ 0 → a * 5^2 + b * 5 + c = 10
  x_intercept : a * (-1)^2 + b * (-1) + c = 0

/-- The x-coordinate of the other x-intercept is 11 -/
theorem other_x_intercept (f : QuadraticFunction) :
  ∃ x : ℝ, x ≠ -1 ∧ f.a * x^2 + f.b * x + f.c = 0 ∧ x = 11 :=
sorry

end other_x_intercept_l4018_401870


namespace ball_return_ways_formula_l4018_401826

/-- The number of ways a ball can return to the starting person after n passes among 7m people. -/
def ball_return_ways (m n : ℕ) : ℚ :=
  (1 / m : ℚ) * ((m - 1 : ℚ)^n + (m - 1 : ℚ) * (-1)^n)

/-- Theorem stating the formula for the number of ways a ball can return to the starting person. -/
theorem ball_return_ways_formula {m n : ℕ} (hm : m ≥ 3) (hn : n ≥ 2) :
  ∃ (c : ℕ → ℚ), c n = ball_return_ways m n :=
by sorry

end ball_return_ways_formula_l4018_401826


namespace range_of_a_l4018_401890

def point_P (a : ℝ) : ℝ × ℝ := (3*a - 9, a + 2)

def on_terminal_side (p : ℝ × ℝ) (α : ℝ) : Prop :=
  (p.1 ≥ 0 ∧ p.2 ≥ 0) ∨ (p.1 ≤ 0 ∧ p.2 ≥ 0) ∨ (p.1 ≤ 0 ∧ p.2 ≤ 0) ∨ (p.1 ≥ 0 ∧ p.2 ≤ 0)

theorem range_of_a (α : ℝ) :
  (∀ a : ℝ, on_terminal_side (point_P a) α ∧ Real.cos α ≤ 0 ∧ Real.sin α > 0) →
  (∀ a : ℝ, a ∈ Set.Ioc (-2) 3) :=
by sorry

end range_of_a_l4018_401890


namespace closest_fraction_to_one_l4018_401872

theorem closest_fraction_to_one : 
  let fractions : List ℚ := [7/8, 8/7, 9/10, 10/11, 11/10]
  ∀ f ∈ fractions, |10/11 - 1| ≤ |f - 1| :=
by
  sorry

end closest_fraction_to_one_l4018_401872


namespace exists_starting_station_l4018_401859

/-- Represents a gasoline station with its fuel amount -/
structure GasStation where
  fuel : ℝ

/-- Represents a circular highway with gasoline stations -/
structure CircularHighway where
  stations : List GasStation
  length : ℝ
  h_positive_length : length > 0

/-- The total fuel available in all stations -/
def total_fuel (highway : CircularHighway) : ℝ :=
  (highway.stations.map (fun s => s.fuel)).sum

/-- Checks if it's possible to complete a lap starting from a given station index -/
def can_complete_lap (highway : CircularHighway) (start_index : ℕ) : Prop :=
  ∃ (direction : Bool), 
    let station_sequence := if direction then 
        highway.stations.rotateLeft start_index
      else 
        (highway.stations.rotateLeft start_index).reverse
    station_sequence.foldl 
      (fun (acc : ℝ) (station : GasStation) => 
        acc + station.fuel - (highway.length / highway.stations.length))
      0 
    ≥ 0

/-- The main theorem to be proved -/
theorem exists_starting_station (highway : CircularHighway) 
  (h_fuel : total_fuel highway = 2 * highway.length) :
  ∃ (i : ℕ), i < highway.stations.length ∧ can_complete_lap highway i :=
sorry

end exists_starting_station_l4018_401859


namespace green_hats_count_l4018_401801

theorem green_hats_count (total_hats : ℕ) (blue_price green_price total_price : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_price = 6)
  (h3 : green_price = 7)
  (h4 : total_price = 530) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_price * blue_hats + green_price * green_hats = total_price ∧
    green_hats = 20 :=
by sorry

end green_hats_count_l4018_401801


namespace lorenzo_thumbtacks_l4018_401833

/-- The number of cans of thumbtacks Lorenzo had -/
def number_of_cans : ℕ := sorry

/-- The number of boards Lorenzo tested -/
def boards_tested : ℕ := 120

/-- The number of tacks remaining in each can at the end of the day -/
def tacks_remaining : ℕ := 30

/-- The total combined number of thumbtacks from the full cans -/
def total_thumbtacks : ℕ := 450

theorem lorenzo_thumbtacks :
  (number_of_cans * (boards_tested + tacks_remaining) = total_thumbtacks) →
  number_of_cans = 3 := by sorry

end lorenzo_thumbtacks_l4018_401833


namespace triangle_proof_l4018_401814

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = t.a * Real.cos t.B ∧ Real.sin t.C = 1/3

-- State the theorem
theorem triangle_proof (t : Triangle) (h : triangle_conditions t) :
  t.A = Real.pi/2 ∧ Real.cos (Real.pi + t.B) = -1/3 := by
  sorry

end triangle_proof_l4018_401814


namespace ellipse_m_value_l4018_401802

/-- Given an ellipse with equation x²/10 + y²/m = 1, foci on the y-axis, and major axis 8, prove that m = 16 -/
theorem ellipse_m_value (x y m : ℝ) : 
  (∀ x y, x^2 / 10 + y^2 / m = 1) →  -- Ellipse equation
  (∃ c, c > 0 ∧ ∀ x, x^2 / 10 + (y + c)^2 / m = 1 ∧ x^2 / 10 + (y - c)^2 / m = 1) →  -- Foci on y-axis
  (∃ y, y^2 / m = 1 ∧ y = 4) →  -- Major axis is 8 (semi-major axis is 4)
  m = 16 := by
sorry


end ellipse_m_value_l4018_401802


namespace percentage_of_x_minus_y_l4018_401886

theorem percentage_of_x_minus_y (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (20 / 100) * (x + y) →
  y = (20 / 100) * x →
  P = 30 := by
sorry

end percentage_of_x_minus_y_l4018_401886


namespace inductive_reasoning_classification_l4018_401812

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| NonInductive

-- Define the types of inductive reasoning
inductive InductiveReasoningType
| Generalization
| Analogy

-- Define a structure for an inference
structure Inference where
  id : Nat
  reasoningType : ReasoningType
  inductiveType : Option InductiveReasoningType

-- Define the inferences
def inference1 : Inference := ⟨1, ReasoningType.Inductive, some InductiveReasoningType.Analogy⟩
def inference2 : Inference := ⟨2, ReasoningType.Inductive, some InductiveReasoningType.Generalization⟩
def inference3 : Inference := ⟨3, ReasoningType.NonInductive, none⟩
def inference4 : Inference := ⟨4, ReasoningType.Inductive, some InductiveReasoningType.Generalization⟩

-- Define a function to check if an inference is inductive
def isInductive (i : Inference) : Prop :=
  i.reasoningType = ReasoningType.Inductive

-- Theorem to prove
theorem inductive_reasoning_classification :
  (isInductive inference1) ∧
  (isInductive inference2) ∧
  (¬isInductive inference3) ∧
  (isInductive inference4) := by
  sorry

end inductive_reasoning_classification_l4018_401812


namespace geometric_sequence_ninth_term_l4018_401856

/-- A geometric sequence with first term 2 and fifth term 4 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) / a n = a 2 / a 1) ∧ a 1 = 2 ∧ a 5 = 4

theorem geometric_sequence_ninth_term (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 9 = 8 := by
  sorry

end geometric_sequence_ninth_term_l4018_401856


namespace unique_solution_for_f_equals_two_l4018_401899

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x - 4
  else if x ≤ 2 then x^2 - 1
  else x/3 + 2

theorem unique_solution_for_f_equals_two :
  ∃! x : ℝ, f x = 2 ∧ x = Real.sqrt 3 := by sorry

end unique_solution_for_f_equals_two_l4018_401899


namespace six_star_three_equals_three_l4018_401828

-- Define the * operation
def star (a b : ℤ) : ℤ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem six_star_three_equals_three : star 6 3 = 3 := by sorry

end six_star_three_equals_three_l4018_401828


namespace trees_planted_total_l4018_401822

/-- Calculates the total number of trees planted given the number of apricot trees and the ratio of peach to apricot trees. -/
def total_trees (apricot_trees : ℕ) (peach_to_apricot_ratio : ℕ) : ℕ :=
  apricot_trees + peach_to_apricot_ratio * apricot_trees

/-- Theorem stating that given the specific conditions, the total number of trees planted is 232. -/
theorem trees_planted_total : total_trees 58 3 = 232 := by
  sorry

#eval total_trees 58 3

end trees_planted_total_l4018_401822


namespace f_zeros_l4018_401869

noncomputable def f (x : ℝ) : ℝ := (1/3) * x - Real.log x

theorem f_zeros (h : ∀ x, x > 0 → f x = (1/3) * x - Real.log x) :
  (∀ x, 1/Real.exp 1 < x ∧ x < 1 → f x ≠ 0) ∧
  (∃ x, 1 < x ∧ x < Real.exp 1 ∧ f x = 0) :=
sorry

end f_zeros_l4018_401869


namespace inequality_holds_iff_x_leq_3_l4018_401849

theorem inequality_holds_iff_x_leq_3 (x : ℕ+) :
  (x + 1 : ℚ) / 3 - (2 * x - 1 : ℚ) / 4 ≥ (x - 3 : ℚ) / 6 ↔ x ≤ 3 := by
  sorry

end inequality_holds_iff_x_leq_3_l4018_401849


namespace ceiling_negative_seven_fourths_cubed_l4018_401898

theorem ceiling_negative_seven_fourths_cubed : ⌈(-7/4)^3⌉ = -5 := by sorry

end ceiling_negative_seven_fourths_cubed_l4018_401898


namespace difference_h_f_l4018_401816

theorem difference_h_f (e f g h : ℕ+) 
  (he : e^5 = f^4)
  (hg : g^3 = h^2)
  (hge : g - e = 31) : 
  h - f = 971 := by sorry

end difference_h_f_l4018_401816


namespace remaining_trees_l4018_401825

/-- Given a park with an initial number of trees, some of which die and others are cut,
    this theorem proves the number of remaining trees. -/
theorem remaining_trees (initial : ℕ) (dead : ℕ) (cut : ℕ) : 
  initial = 86 → dead = 15 → cut = 23 → initial - (dead + cut) = 48 := by
  sorry

end remaining_trees_l4018_401825


namespace parametric_equations_represent_line_l4018_401848

/-- Proves that the given parametric equations represent the straight line 2x - y + 1 = 0 -/
theorem parametric_equations_represent_line :
  ∀ (t : ℝ), 2 * (1 - t) - (3 - 2*t) + 1 = 0 := by
  sorry

#check parametric_equations_represent_line

end parametric_equations_represent_line_l4018_401848


namespace min_value_fraction_l4018_401895

theorem min_value_fraction (x : ℝ) (h : x ≥ 0) :
  (5 * x^2 + 20 * x + 25) / (8 * (1 + x)) ≥ 65 / 16 ∧
  ∃ y : ℝ, y ≥ 0 ∧ (5 * y^2 + 20 * y + 25) / (8 * (1 + y)) = 65 / 16 :=
by sorry

end min_value_fraction_l4018_401895


namespace matrix_sum_proof_l4018_401819

theorem matrix_sum_proof :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, 3; -2, 1]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-1, 5; 8, -3]
  A + B = !![3, 8; 6, -2] := by
  sorry

end matrix_sum_proof_l4018_401819


namespace range_of_a_l4018_401838

/-- Given propositions p and q, where p is a necessary but not sufficient condition for q,
    prove that the range of real number a is [-1/2, 1]. -/
theorem range_of_a (x a : ℝ) : 
  (∀ x, x^2 - ax - 2*a^2 < 0 → x^2 - 2*x - 3 < 0) ∧ 
  (∃ x, x^2 - 2*x - 3 < 0 ∧ x^2 - ax - 2*a^2 ≥ 0) →
  -1/2 ≤ a ∧ a ≤ 1 :=
by sorry

end range_of_a_l4018_401838


namespace meaningful_sqrt_over_x_l4018_401878

theorem meaningful_sqrt_over_x (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / x) ↔ x ≥ -3 ∧ x ≠ 0 :=
by sorry

end meaningful_sqrt_over_x_l4018_401878


namespace sam_distance_l4018_401875

theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
  (h1 : marguerite_distance = 150)
  (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) : 
  (marguerite_distance / marguerite_time) * sam_time = 200 := by
  sorry

end sam_distance_l4018_401875


namespace tan_equality_implies_specific_angles_l4018_401860

theorem tan_equality_implies_specific_angles (m : ℤ) :
  -180 < m ∧ m < 180 →
  Real.tan (↑m * π / 180) = Real.tan (405 * π / 180) →
  m = 45 ∨ m = -135 := by
sorry

end tan_equality_implies_specific_angles_l4018_401860


namespace tuesday_flower_sales_ratio_l4018_401836

/-- Represents the number of flowers sold -/
structure FlowerSales where
  roses : ℕ
  lilacs : ℕ
  gardenias : ℕ

/-- Represents the ratio of two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of roses to lilacs -/
def roseToLilacRatio (sales : FlowerSales) : Ratio :=
  { numerator := sales.roses, denominator := sales.lilacs }

theorem tuesday_flower_sales_ratio : 
  ∀ (sales : FlowerSales), 
    sales.lilacs = 10 →
    sales.gardenias = sales.lilacs / 2 →
    sales.roses + sales.lilacs + sales.gardenias = 45 →
    (roseToLilacRatio sales).numerator = 3 ∧ (roseToLilacRatio sales).denominator = 1 := by
  sorry


end tuesday_flower_sales_ratio_l4018_401836


namespace nina_money_problem_l4018_401845

theorem nina_money_problem (x : ℚ) :
  (5 * x = 8 * (x - 1.25)) → (5 * x = 50 / 3) := by
  sorry

end nina_money_problem_l4018_401845


namespace expansion_properties_l4018_401804

open Real Nat

/-- Represents the expansion of (1 + 2∛x)^n -/
def expansion (n : ℕ) (x : ℝ) := (1 + 2 * x^(1/3))^n

/-- Coefficient of the r-th term in the expansion -/
def coefficient (n r : ℕ) : ℝ := 2^r * choose n r

/-- Condition for the coefficient relation -/
def coefficient_condition (n : ℕ) : Prop :=
  ∃ r, coefficient n r = 2 * coefficient n (r-1) ∧
       coefficient n r = 5/6 * coefficient n (r+1)

/-- Sum of all coefficients in the expansion -/
def sum_coefficients (n : ℕ) : ℝ := 3^n

/-- Sum of all binomial coefficients -/
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

/-- Rational terms in the expansion -/
def rational_terms (n : ℕ) : List (ℝ × ℕ) :=
  [(1, 0), (560, 1), (448, 2), (2016, 3)]

theorem expansion_properties (n : ℕ) :
  coefficient_condition n →
  n = 7 ∧
  sum_coefficients n = 2187 ∧
  sum_binomial_coefficients n = 128 ∧
  rational_terms n = [(1, 0), (560, 1), (448, 2), (2016, 3)] :=
by sorry

end expansion_properties_l4018_401804


namespace shopkeeper_profit_percentage_l4018_401818

/-- Calculates the profit percentage for a shopkeeper who sold 30 articles at the cost price of 35 articles -/
theorem shopkeeper_profit_percentage :
  let articles_sold : ℕ := 30
  let cost_price_articles : ℕ := 35
  let profit_ratio : ℚ := (cost_price_articles - articles_sold) / articles_sold
  profit_ratio * 100 = 5 / 30 * 100 := by
sorry

end shopkeeper_profit_percentage_l4018_401818


namespace unique_prime_pair_divisibility_l4018_401879

theorem unique_prime_pair_divisibility : 
  ∃! (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (3 * p^(q-1) + 1) ∣ (11^p + 17^p) ∧
    p = 3 ∧ q = 3 := by
  sorry

end unique_prime_pair_divisibility_l4018_401879


namespace second_win_proof_l4018_401831

/-- Represents the financial transactions of a man and calculates the amount won in the second round --/
def calculate_second_win (initial_amount : ℚ) (first_win : ℚ) : ℚ :=
  let after_first_loss := initial_amount * (2/3)
  let after_first_win := after_first_loss + first_win
  let after_second_loss := after_first_win * (2/3)
  initial_amount - after_second_loss

/-- Proves that the calculated second win amount results in the initial amount --/
theorem second_win_proof (initial_amount : ℚ) (first_win : ℚ) :
  let second_win := calculate_second_win initial_amount first_win
  let final_amount := (((initial_amount * (2/3) + first_win) * (2/3)) + second_win)
  initial_amount = 48.00000000000001 ∧ first_win = 10 →
  final_amount = initial_amount ∧ second_win = 20 := by
  sorry

#eval calculate_second_win 48.00000000000001 10

end second_win_proof_l4018_401831


namespace cube_edge_length_l4018_401880

theorem cube_edge_length (x : ℝ) : x > 0 → 6 * x^2 = 1014 → x = 13 := by
  sorry

end cube_edge_length_l4018_401880


namespace outer_circle_radius_l4018_401847

/-- Given a circular race track with an inner circumference of 440 meters and a width of 14 meters,
    the radius of the outer circle is equal to (440 / (2 * π)) + 14. -/
theorem outer_circle_radius (inner_circumference : ℝ) (track_width : ℝ)
    (h1 : inner_circumference = 440)
    (h2 : track_width = 14) :
    (inner_circumference / (2 * Real.pi) + track_width) = (440 / (2 * Real.pi) + 14) := by
  sorry

end outer_circle_radius_l4018_401847


namespace smallest_solution_of_quadratic_l4018_401893

theorem smallest_solution_of_quadratic : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 10*x - 24
  ∃ (x : ℝ), f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = -12 := by
  sorry

end smallest_solution_of_quadratic_l4018_401893


namespace george_room_painting_choices_l4018_401827

theorem george_room_painting_choices :
  (Nat.choose 10 3) * 5 = 600 := by sorry

end george_room_painting_choices_l4018_401827


namespace total_earned_is_144_l4018_401854

/-- Calculates the total money earned from selling milk and butter --/
def total_money_earned (milk_price : ℚ) (butter_conversion : ℚ) (butter_price : ℚ) 
  (num_cows : ℕ) (milk_per_cow : ℚ) (num_customers : ℕ) (milk_per_customer : ℚ) : ℚ :=
  let total_milk := num_cows * milk_per_cow
  let sold_milk := min total_milk (num_customers * milk_per_customer)
  let remaining_milk := total_milk - sold_milk
  let butter_sticks := remaining_milk * butter_conversion
  milk_price * sold_milk + butter_price * butter_sticks

/-- Theorem stating that the total money earned is $144 given the problem conditions --/
theorem total_earned_is_144 :
  total_money_earned 3 2 (3/2) 12 4 6 6 = 144 := by
  sorry

end total_earned_is_144_l4018_401854


namespace total_salary_is_583_l4018_401873

/-- The total amount paid to two employees per week, given their relative salaries -/
def total_salary (n_salary : ℝ) : ℝ :=
  n_salary + 1.2 * n_salary

/-- Proof that the total salary for two employees is $583 per week -/
theorem total_salary_is_583 :
  total_salary 265 = 583 := by
  sorry

end total_salary_is_583_l4018_401873


namespace ocean_area_scientific_notation_l4018_401841

theorem ocean_area_scientific_notation : 
  361000000 = 3.61 * (10 ^ 8) := by sorry

end ocean_area_scientific_notation_l4018_401841


namespace game_points_difference_l4018_401824

theorem game_points_difference (eric_points mark_points samanta_points : ℕ) : 
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points > mark_points →
  samanta_points + mark_points + eric_points = 32 →
  samanta_points - mark_points = 8 :=
by sorry

end game_points_difference_l4018_401824


namespace youngest_child_age_proof_l4018_401842

def youngest_child_age (n : ℕ) (interval : ℕ) (total_age : ℕ) : ℕ :=
  (total_age - (n - 1) * n * interval / 2) / n

theorem youngest_child_age_proof (n : ℕ) (interval : ℕ) (total_age : ℕ) 
  (h1 : n = 5)
  (h2 : interval = 3)
  (h3 : total_age = 50)
  (h4 : youngest_child_age n interval total_age * 2 = youngest_child_age n interval total_age + (n - 1) * interval) :
  youngest_child_age n interval total_age = 4 := by
sorry

#eval youngest_child_age 5 3 50

end youngest_child_age_proof_l4018_401842


namespace system_solution_l4018_401810

theorem system_solution (x y : ℝ) : 
  (x + y = 1 ∧ x - y = 3) → (x = 2 ∧ y = -1) :=
by sorry

end system_solution_l4018_401810


namespace factorial_345_trailing_zeros_l4018_401834

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_345_trailing_zeros :
  trailingZeros 345 = 84 :=
sorry

end factorial_345_trailing_zeros_l4018_401834


namespace problem_factory_daily_production_l4018_401843

/-- A factory that produces toys -/
structure ToyFactory where
  weekly_production : ℕ
  working_days : ℕ
  daily_production : ℕ
  h1 : weekly_production = working_days * daily_production

/-- The specific toy factory in the problem -/
def problem_factory : ToyFactory where
  weekly_production := 8000
  working_days := 4
  daily_production := 2000
  h1 := rfl

/-- Theorem stating that the daily production of the problem factory is 2000 toys -/
theorem problem_factory_daily_production :
  problem_factory.daily_production = 2000 := by sorry

end problem_factory_daily_production_l4018_401843


namespace binomial_expectation_and_variance_l4018_401889

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  ξ : ℝ → ℝ  -- The random variable

/-- The expected value of a random variable -/
def expectation (X : ℝ → ℝ) : ℝ := sorry

/-- The variance of a random variable -/
def variance (X : ℝ → ℝ) : ℝ := sorry

theorem binomial_expectation_and_variance 
  (ξ : BinomialDistribution 5 (1/2)) 
  (η : ℝ → ℝ) 
  (h : η = λ x => 5 * ξ.ξ x) : 
  expectation η = 25/2 ∧ variance η = 125/4 := by sorry

end binomial_expectation_and_variance_l4018_401889


namespace polynomial_multiplication_l4018_401862

theorem polynomial_multiplication (y : ℝ) :
  (3*y - 2 + 4) * (2*y^12 + 3*y^11 - y^9 - y^8) =
  6*y^13 + 13*y^12 + 6*y^11 - 3*y^10 - 5*y^9 - 2*y^8 := by
  sorry

end polynomial_multiplication_l4018_401862


namespace friendship_subset_exists_l4018_401882

/-- Represents a friendship relation between students -/
def FriendshipRelation (S : Type) := S → S → Prop

/-- A school is valid if it satisfies the friendship condition -/
def ValidSchool (S : Type) (friendship : FriendshipRelation S) (students : Finset S) : Prop :=
  ∀ s ∈ students, ∃ t ∈ students, s ≠ t ∧ friendship s t

theorem friendship_subset_exists 
  (S : Type) 
  (friendship : FriendshipRelation S) 
  (students : Finset S) 
  (h_valid : ValidSchool S friendship students)
  (h_count : students.card = 101) :
  ∀ n : ℕ, 1 < n → n < 101 → 
    ∃ subset : Finset S, subset.card = n ∧ subset ⊆ students ∧
      ∀ s ∈ subset, ∃ t ∈ subset, s ≠ t ∧ friendship s t :=
by
  sorry


end friendship_subset_exists_l4018_401882


namespace f_x_plus_3_l4018_401867

/-- Given a function f: ℝ → ℝ defined as f(x) = x^2 for all real numbers x,
    prove that f(x + 3) = x^2 + 6x + 9 for all real numbers x. -/
theorem f_x_plus_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x^2) :
  ∀ x : ℝ, f (x + 3) = x^2 + 6*x + 9 := by
  sorry

end f_x_plus_3_l4018_401867


namespace blue_marbles_count_l4018_401858

theorem blue_marbles_count (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) 
  (h1 : total = 20)
  (h2 : red = 9)
  (h3 : prob_red_or_white = 3/4) :
  ∃ blue : ℕ, blue = 5 ∧ 
    (blue + red : ℚ) / total + prob_red_or_white = 1 := by
  sorry

end blue_marbles_count_l4018_401858


namespace sam_current_age_l4018_401823

/-- Sam's current age -/
def sam_age : ℕ := 46

/-- Drew's current age -/
def drew_age : ℕ := 12

/-- Theorem stating Sam's current age is 46, given the conditions -/
theorem sam_current_age :
  (sam_age + 5 = 3 * (drew_age + 5)) → sam_age = 46 := by
  sorry

end sam_current_age_l4018_401823


namespace sets_with_property_P_l4018_401891

-- Define property P
def property_P (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ) (k : ℝ), (x, y) ∈ M → 0 < k → k < 1 → (k * x, k * y) ∈ M

-- Define the four sets
def set1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 ≥ p.2}
def set2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 < 1}
def set3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * p.1 + 2 * p.2 = 0}
def set4 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^3 + p.2^3 - p.1^2 * p.2 = 0}

-- Theorem stating which sets possess property P
theorem sets_with_property_P :
  property_P set2 ∧ property_P set4 ∧ ¬property_P set1 ∧ ¬property_P set3 := by
  sorry

end sets_with_property_P_l4018_401891


namespace janine_read_five_books_last_month_l4018_401840

/-- The number of books Janine read last month -/
def last_month_books : ℕ := sorry

/-- The number of books Janine read this month -/
def this_month_books : ℕ := 2 * last_month_books

/-- The number of pages in each book -/
def pages_per_book : ℕ := 10

/-- The total number of pages Janine read in two months -/
def total_pages : ℕ := 150

theorem janine_read_five_books_last_month :
  last_month_books = 5 :=
by sorry

end janine_read_five_books_last_month_l4018_401840


namespace a_finishes_in_eight_days_l4018_401863

/-- Given two workers A and B who can finish a job together in a certain number of days,
    this function calculates how long it takes for A to finish the job alone. -/
def time_for_a_alone (total_time_together : ℚ) (days_worked_together : ℚ) (days_a_alone : ℚ) : ℚ :=
  let work_rate_together := 1 / total_time_together
  let work_done_together := work_rate_together * days_worked_together
  let remaining_work := 1 - work_done_together
  let work_rate_a := remaining_work / days_a_alone
  1 / work_rate_a

/-- Theorem stating that under the given conditions, A can finish the job alone in 8 days. -/
theorem a_finishes_in_eight_days :
  time_for_a_alone 40 10 6 = 8 := by sorry

end a_finishes_in_eight_days_l4018_401863


namespace inequality_solution_set_min_mn_value_l4018_401876

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem inequality_solution_set (x : ℝ) :
  (f 1 x ≥ 4 - |x + 1|) ↔ (x ≤ -2 ∨ x ≥ 2) := by sorry

theorem min_mn_value (a m n : ℝ) :
  (∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  m > 0 →
  n > 0 →
  1/m + 1/(2*n) = a →
  ∀ k, m*n ≤ k → 2 ≤ k := by sorry

end inequality_solution_set_min_mn_value_l4018_401876


namespace valid_student_counts_exists_valid_distributions_l4018_401864

/-- Represents the distribution of students in groups -/
structure StudentDistribution where
  total_groups : ℕ
  groups_with_13 : ℕ
  total_students : ℕ

/-- Checks if a given distribution satisfies the problem conditions -/
def is_valid_distribution (d : StudentDistribution) : Prop :=
  d.total_groups = 6 ∧
  d.groups_with_13 = 4 ∧
  (d.total_students = 76 ∨ d.total_students = 80)

/-- Theorem stating the only valid total numbers of students -/
theorem valid_student_counts :
  ∀ d : StudentDistribution,
    is_valid_distribution d →
    (d.total_students = 76 ∨ d.total_students = 80) :=
by
  sorry

/-- Theorem proving the existence of valid distributions -/
theorem exists_valid_distributions :
  ∃ d₁ d₂ : StudentDistribution,
    is_valid_distribution d₁ ∧
    is_valid_distribution d₂ ∧
    d₁.total_students = 76 ∧
    d₂.total_students = 80 :=
by
  sorry

end valid_student_counts_exists_valid_distributions_l4018_401864
