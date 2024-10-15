import Mathlib

namespace NUMINAMATH_CALUDE_circle_to_octagon_area_ratio_l3669_366946

/-- The ratio of the area of a circle inscribed in a regular octagon to the area of the octagon,
    where the circle's radius reaches the midpoint of each octagon side. -/
theorem circle_to_octagon_area_ratio : ∃ (a b : ℕ), 
  (a : ℝ).sqrt / b * π = π / (4 * (1 + Real.sqrt 2)) ∧ a * b = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_to_octagon_area_ratio_l3669_366946


namespace NUMINAMATH_CALUDE_floor_sum_example_l3669_366949

theorem floor_sum_example : ⌊(24.7 : ℝ)⌋ + ⌊(-24.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3669_366949


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3669_366995

theorem simplify_sqrt_expression : 
  Real.sqrt 80 - 3 * Real.sqrt 10 + (2 * Real.sqrt 500) / Real.sqrt 5 = Real.sqrt 2205 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3669_366995


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l3669_366954

theorem binomial_expansion_problem (a b : ℝ) (n : ℕ) :
  (∀ k, 1 ≤ k ∧ k ≤ n + 1 → Nat.choose n (k - 1) ≤ Nat.choose n 5) ∧
  (a + b = 4) →
  (n = 10) ∧
  ((4^n + 7) % 3 = 2) := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l3669_366954


namespace NUMINAMATH_CALUDE_student_sums_correct_l3669_366920

theorem student_sums_correct (total_sums : ℕ) (wrong_ratio : ℕ) 
  (h1 : total_sums = 48) 
  (h2 : wrong_ratio = 2) : 
  ∃ (correct_sums : ℕ), 
    correct_sums + wrong_ratio * correct_sums = total_sums ∧ 
    correct_sums = 16 := by
  sorry

end NUMINAMATH_CALUDE_student_sums_correct_l3669_366920


namespace NUMINAMATH_CALUDE_comic_books_bought_correct_comic_books_bought_l3669_366957

theorem comic_books_bought (initial : ℕ) (current : ℕ) : ℕ :=
  let sold := initial / 2
  let remaining := initial - sold
  let bought := current - remaining
  bought

theorem correct_comic_books_bought :
  comic_books_bought 22 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_bought_correct_comic_books_bought_l3669_366957


namespace NUMINAMATH_CALUDE_books_and_students_count_l3669_366965

/-- The number of books distributed to students -/
def total_books : ℕ := 26

/-- The number of students receiving books -/
def total_students : ℕ := 6

/-- Condition 1: If each person receives 3 books, there will be 8 books left -/
axiom condition1 : total_books = 3 * total_students + 8

/-- Condition 2: If each of the previous students receives 5 books, 
    then the last person will not receive 2 books -/
axiom condition2 : 
  total_books - 5 * (total_students - 1) < 2 ∧ 
  total_books - 5 * (total_students - 1) ≥ 0

/-- Theorem: Given the conditions, prove that the number of books is 26 
    and the number of students is 6 -/
theorem books_and_students_count : 
  total_books = 26 ∧ total_students = 6 := by
  sorry

end NUMINAMATH_CALUDE_books_and_students_count_l3669_366965


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l3669_366902

/-- Calculates the total revenue from concert ticket sales --/
theorem concert_ticket_revenue : 
  let ticket_price : ℕ := 20
  let first_discount : ℕ := 40
  let second_discount : ℕ := 15
  let first_group : ℕ := 10
  let second_group : ℕ := 20
  let total_tickets : ℕ := 50
  
  let first_group_revenue := first_group * (ticket_price - (ticket_price * first_discount / 100))
  let second_group_revenue := second_group * (ticket_price - (ticket_price * second_discount / 100))
  let full_price_revenue := (total_tickets - first_group - second_group) * ticket_price
  
  first_group_revenue + second_group_revenue + full_price_revenue = 860 :=
by
  sorry


end NUMINAMATH_CALUDE_concert_ticket_revenue_l3669_366902


namespace NUMINAMATH_CALUDE_ellipse_properties_l3669_366913

-- Define the ellipse C
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y m : ℝ) : Prop :=
  y = x + m

-- Define the theorem
theorem ellipse_properties
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0)
  (h_triangle : ∃ A F₁ F₂ : ℝ × ℝ, 
    ellipse A.1 A.2 a b ∧
    ellipse F₁.1 F₁.2 a b ∧
    ellipse F₂.1 F₂.2 a b ∧
    (A.2 - F₁.2)^2 + (A.1 - F₁.1)^2 = (A.2 - F₂.2)^2 + (A.1 - F₂.1)^2 ∧
    (F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2 = 8) :
  (∀ x y, ellipse x y 2 (Real.sqrt 2)) ∧
  (∀ P Q : ℝ × ℝ, 
    ellipse P.1 P.2 2 (Real.sqrt 2) ∧
    ellipse Q.1 Q.2 2 (Real.sqrt 2) ∧
    line P.1 P.2 1 ∧
    line Q.1 Q.2 1 →
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 80/9) ∧
  (∀ m : ℝ, 
    (∃ P Q : ℝ × ℝ,
      ellipse P.1 P.2 2 (Real.sqrt 2) ∧
      ellipse Q.1 Q.2 2 (Real.sqrt 2) ∧
      line P.1 P.2 m ∧
      line Q.1 Q.2 m ∧
      P.1 * Q.2 - P.2 * Q.1 = 8/3) ↔
    (m = 2 ∨ m = -2 ∨ m = Real.sqrt 2 ∨ m = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3669_366913


namespace NUMINAMATH_CALUDE_circles_intersect_l3669_366905

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 8*p.2 - 8 = 0}

/-- Circle C₂ with equation x² + y² - 4x - 5 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 5 = 0}

/-- The center of circle C₁ -/
def center_C₁ : ℝ × ℝ := (-1, -4)

/-- The radius of circle C₁ -/
def radius_C₁ : ℝ := 5

/-- The center of circle C₂ -/
def center_C₂ : ℝ × ℝ := (2, 0)

/-- The radius of circle C₂ -/
def radius_C₂ : ℝ := 3

/-- Theorem stating that circles C₁ and C₂ are intersecting -/
theorem circles_intersect : ∃ p : ℝ × ℝ, p ∈ C₁ ∩ C₂ := by sorry

end NUMINAMATH_CALUDE_circles_intersect_l3669_366905


namespace NUMINAMATH_CALUDE_f_continuous_iff_l3669_366974

noncomputable def f (b c x : ℝ) : ℝ :=
  if x ≤ 4 then 2 * x^2 + 5
  else if x ≤ 6 then b * x + 3
  else c * x^2 - 2 * x + 9

theorem f_continuous_iff (b c : ℝ) :
  Continuous (f b c) ↔ b = 8.5 ∧ c = 19/12 := by sorry

end NUMINAMATH_CALUDE_f_continuous_iff_l3669_366974


namespace NUMINAMATH_CALUDE_worker_completion_time_l3669_366936

theorem worker_completion_time (worker_b_time worker_ab_time : Real) 
  (hb : worker_b_time = 10)
  (hab : worker_ab_time = 3.333333333333333)
  : ∃ worker_a_time : Real, 
    worker_a_time = 5 ∧ 
    1 / worker_a_time + 1 / worker_b_time = 1 / worker_ab_time :=
by
  sorry

end NUMINAMATH_CALUDE_worker_completion_time_l3669_366936


namespace NUMINAMATH_CALUDE_visual_range_increase_l3669_366918

theorem visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 100)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_visual_range_increase_l3669_366918


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l3669_366924

theorem shaded_area_between_circles (r : ℝ) : 
  r > 0 → 
  (2 * r = 6) → 
  (π * (3 * r)^2 - π * r^2 = 72 * π) := by
sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l3669_366924


namespace NUMINAMATH_CALUDE_simplify_expression_l3669_366983

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 9) - (x + 6)*(3*x + 2) = 3*x - 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3669_366983


namespace NUMINAMATH_CALUDE_china_first_negative_numbers_l3669_366982

-- Define an enumeration for the countries
inductive Country
  | France
  | China
  | England
  | UnitedStates

-- Define a function that represents the property of being the first country to recognize and use negative numbers
def firstToUseNegativeNumbers : Country → Prop :=
  fun c => c = Country.China

-- Theorem statement
theorem china_first_negative_numbers :
  ∃ c : Country, firstToUseNegativeNumbers c ∧
  (c = Country.France ∨ c = Country.China ∨ c = Country.England ∨ c = Country.UnitedStates) :=
by
  sorry


end NUMINAMATH_CALUDE_china_first_negative_numbers_l3669_366982


namespace NUMINAMATH_CALUDE_total_hamburger_combinations_l3669_366948

/-- The number of condiments available for hamburgers. -/
def num_condiments : ℕ := 8

/-- The number of options for meat patties. -/
def num_patty_options : ℕ := 4

/-- Calculates the number of possible condiment combinations. -/
def condiment_combinations : ℕ := 2^num_condiments

/-- Theorem stating the total number of different hamburger combinations. -/
theorem total_hamburger_combinations : 
  num_patty_options * condiment_combinations = 1024 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburger_combinations_l3669_366948


namespace NUMINAMATH_CALUDE_shaded_area_square_minus_semicircles_l3669_366932

/-- The area of a square with side length 14 cm minus the area of two semicircles 
    with diameters equal to the side length of the square is equal to 196 - 49π cm². -/
theorem shaded_area_square_minus_semicircles : 
  let side_length : ℝ := 14
  let square_area : ℝ := side_length ^ 2
  let semicircle_radius : ℝ := side_length / 2
  let semicircles_area : ℝ := π * semicircle_radius ^ 2
  square_area - semicircles_area = 196 - 49 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_square_minus_semicircles_l3669_366932


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3669_366991

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 - Complex.I) = 3 + Complex.I) : 
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3669_366991


namespace NUMINAMATH_CALUDE_ninth_group_number_l3669_366901

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  totalWorkers : ℕ
  sampleSize : ℕ
  samplingInterval : ℕ
  fifthGroupNumber : ℕ

/-- Calculates the number drawn from a specific group given the sampling parameters -/
def groupNumber (s : SystematicSampling) (groupIndex : ℕ) : ℕ :=
  s.fifthGroupNumber + (groupIndex - 5) * s.samplingInterval

/-- Theorem stating that for the given systematic sampling scenario, 
    the number drawn from the 9th group is 43 -/
theorem ninth_group_number (s : SystematicSampling) 
  (h1 : s.totalWorkers = 100)
  (h2 : s.sampleSize = 20)
  (h3 : s.samplingInterval = 5)
  (h4 : s.fifthGroupNumber = 23) :
  groupNumber s 9 = 43 := by
  sorry

end NUMINAMATH_CALUDE_ninth_group_number_l3669_366901


namespace NUMINAMATH_CALUDE_expression_equality_l3669_366985

theorem expression_equality : 
  |2 - Real.sqrt 3| - (2022 - Real.pi)^0 + Real.sqrt 12 = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3669_366985


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l3669_366968

theorem magnitude_of_complex_fraction : Complex.abs (1 / (1 - 2 * Complex.I)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l3669_366968


namespace NUMINAMATH_CALUDE_sum_proper_divisors_256_l3669_366919

theorem sum_proper_divisors_256 : 
  (Finset.filter (fun d => d ≠ 256 ∧ 256 % d = 0) (Finset.range 257)).sum id = 255 := by
  sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_256_l3669_366919


namespace NUMINAMATH_CALUDE_faster_runner_overtakes_in_five_laps_l3669_366935

/-- The length of the track in meters -/
def track_length : ℝ := 400

/-- The speed ratio of the faster runner to the slower runner -/
def speed_ratio : ℝ := 1.25

/-- The number of laps after which the faster runner overtakes the slower runner -/
def overtake_laps : ℝ := 5

/-- Theorem stating that the faster runner overtakes the slower runner after 5 laps -/
theorem faster_runner_overtakes_in_five_laps :
  ∀ (v : ℝ), v > 0 →
  speed_ratio * v * (overtake_laps * track_length / v) =
  (overtake_laps + 1) * track_length :=
by sorry

end NUMINAMATH_CALUDE_faster_runner_overtakes_in_five_laps_l3669_366935


namespace NUMINAMATH_CALUDE_units_digit_17_cubed_times_24_l3669_366996

theorem units_digit_17_cubed_times_24 : (17^3 * 24) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_cubed_times_24_l3669_366996


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l3669_366953

/-- Given a line segment with one endpoint (5,4) and midpoint (3.5,10.5),
    the sum of the coordinates of the other endpoint is 19. -/
theorem endpoint_coordinate_sum : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
  fun x₁ y₁ x_mid y_mid x₂ y₂ =>
    x₁ = 5 ∧ y₁ = 4 ∧ x_mid = 3.5 ∧ y_mid = 10.5 ∧
    x_mid = (x₁ + x₂) / 2 ∧ y_mid = (y₁ + y₂) / 2 →
    x₂ + y₂ = 19

/-- Proof of the theorem -/
theorem endpoint_coordinate_sum_proof : 
  ∃ x₂ y₂, endpoint_coordinate_sum 5 4 3.5 10.5 x₂ y₂ := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l3669_366953


namespace NUMINAMATH_CALUDE_candy_game_bounds_l3669_366973

/-- Represents the colors of candies -/
inductive Color
  | Yellow
  | Red
  | Green
  | Blue

/-- Represents a collection of candies -/
structure CandyCollection :=
  (yellow : ℕ)
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- Represents the game state -/
structure GameState :=
  (remaining : CandyCollection)
  (jia : CandyCollection)
  (yi : CandyCollection)

def total_candies (c : CandyCollection) : ℕ :=
  c.yellow + c.red + c.green + c.blue

/-- Yi's turn: takes two candies (or the last one if only one is left) -/
def yi_turn (state : GameState) : GameState :=
  sorry

/-- Jia's turn: takes one candy of each color from the remaining candies -/
def jia_turn (state : GameState) : GameState :=
  sorry

/-- Plays the game until all candies are taken -/
def play_game (initial_state : GameState) : GameState :=
  sorry

theorem candy_game_bounds :
  ∀ (initial : CandyCollection),
    total_candies initial = 22 →
    initial.yellow ≥ initial.red ∧
    initial.yellow ≥ initial.green ∧
    initial.yellow ≥ initial.blue →
    let final_state := play_game { remaining := initial, jia := ⟨0,0,0,0⟩, yi := ⟨0,0,0,0⟩ }
    total_candies final_state.jia = total_candies final_state.yi →
    8 ≤ initial.yellow ∧ initial.yellow ≤ 16 :=
  sorry

end NUMINAMATH_CALUDE_candy_game_bounds_l3669_366973


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3669_366945

def A : Set ℤ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℤ := {-2, -1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3669_366945


namespace NUMINAMATH_CALUDE_triangle_max_height_l3669_366986

/-- In a triangle ABC with sides a, b, c corresponding to angles A, B, C respectively,
    given that c = 1 and a*cos(B) + b*cos(A) = 2*cos(C),
    the maximum value of the height h on side AB is √3/2 -/
theorem triangle_max_height (a b c : ℝ) (A B C : ℝ) (h : ℝ) :
  c = 1 →
  a * Real.cos B + b * Real.cos A = 2 * Real.cos C →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  h ≤ Real.sqrt 3 / 2 ∧ ∃ (a' b' : ℝ), h = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_height_l3669_366986


namespace NUMINAMATH_CALUDE_student_average_grade_l3669_366926

theorem student_average_grade 
  (courses_last_year : ℕ)
  (courses_year_before : ℕ)
  (avg_grade_year_before : ℚ)
  (avg_grade_two_years : ℚ)
  (h1 : courses_last_year = 6)
  (h2 : courses_year_before = 5)
  (h3 : avg_grade_year_before = 50)
  (h4 : avg_grade_two_years = 77)
  : ∃ (avg_grade_last_year : ℚ), avg_grade_last_year = 99.5 := by
  sorry

end NUMINAMATH_CALUDE_student_average_grade_l3669_366926


namespace NUMINAMATH_CALUDE_eighth_term_is_sixteen_l3669_366907

def odd_term (n : ℕ) : ℕ := 2 * n - 1

def even_term (n : ℕ) : ℕ := 4 * n

def sequence_term (n : ℕ) : ℕ :=
  if n % 2 = 1 then odd_term ((n + 1) / 2) else even_term (n / 2)

theorem eighth_term_is_sixteen : sequence_term 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_sixteen_l3669_366907


namespace NUMINAMATH_CALUDE_greatest_y_value_l3669_366988

theorem greatest_y_value (y : ℕ) (h1 : y > 0) (h2 : ∃ k : ℕ, y = 4 * k) (h3 : y^3 < 8000) :
  y ≤ 16 ∧ ∃ (y' : ℕ), y' = 16 ∧ ∃ (k : ℕ), y' = 4 * k ∧ y'^3 < 8000 :=
sorry

end NUMINAMATH_CALUDE_greatest_y_value_l3669_366988


namespace NUMINAMATH_CALUDE_no_prime_multiples_of_ten_in_range_l3669_366950

theorem no_prime_multiples_of_ten_in_range : 
  ¬ ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 10000 ∧ 10 ∣ n ∧ Nat.Prime n ∧ n > 10 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_multiples_of_ten_in_range_l3669_366950


namespace NUMINAMATH_CALUDE_binomial_20_4_l3669_366939

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_4_l3669_366939


namespace NUMINAMATH_CALUDE_owls_joined_l3669_366900

def initial_owls : ℕ := 3
def final_owls : ℕ := 5

theorem owls_joined : final_owls - initial_owls = 2 := by
  sorry

end NUMINAMATH_CALUDE_owls_joined_l3669_366900


namespace NUMINAMATH_CALUDE_function_inequality_solution_l3669_366970

theorem function_inequality_solution (f g h : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, (x - y) * (f x) + h x - x * y + y^2 ≤ h y)
  (h2 : ∀ x y : ℝ, h y ≤ (x - y) * (g x) + h x - x * y + y^2) :
  ∃ a b : ℝ, 
    (∀ x : ℝ, f x = -x + a) ∧ 
    (∀ x : ℝ, g x = -x + a) ∧ 
    (∀ x : ℝ, h x = x^2 - a*x + b) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_solution_l3669_366970


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_count_l3669_366994

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
def five_balls_three_boxes : ℕ := distribute_balls 5 3

theorem five_balls_three_boxes_count : five_balls_three_boxes = 21 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_count_l3669_366994


namespace NUMINAMATH_CALUDE_four_foldable_positions_l3669_366967

/-- Represents a position where an additional square can be attached --/
inductive Position
| Top
| TopRight
| Right
| BottomRight
| Bottom
| BottomLeft
| Left
| TopLeft
| CenterTop
| CenterRight
| CenterBottom
| CenterLeft

/-- Represents the cross-shaped polygon --/
structure CrossPolygon :=
  (squares : Fin 5 → Unit)

/-- Represents the resulting polygon after attaching an additional square --/
structure ResultingPolygon :=
  (base : CrossPolygon)
  (additional : Position)

/-- Predicate to check if a resulting polygon can be folded into a cube with one face missing --/
def can_fold_to_cube (p : ResultingPolygon) : Prop :=
  sorry

/-- The main theorem --/
theorem four_foldable_positions :
  ∃ (valid_positions : Finset Position),
    (valid_positions.card = 4) ∧
    (∀ p : Position, p ∈ valid_positions ↔ 
      can_fold_to_cube ⟨CrossPolygon.mk (λ _ => Unit.unit), p⟩) :=
  sorry

end NUMINAMATH_CALUDE_four_foldable_positions_l3669_366967


namespace NUMINAMATH_CALUDE_train_distance_problem_l3669_366962

/-- The distance between two points P and Q, given the conditions of two trains traveling towards each other --/
theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 50) (h2 : v2 = 40) (h3 : d = 100) : 
  v1 * (d / (v1 - v2) + d / v2) = 900 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l3669_366962


namespace NUMINAMATH_CALUDE_calculate_expression_l3669_366947

theorem calculate_expression : 
  |-Real.sqrt 3| - (4 - Real.pi)^0 - 2 * Real.sin (60 * π / 180) + (1/5)⁻¹ = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3669_366947


namespace NUMINAMATH_CALUDE_fraction_equality_l3669_366987

theorem fraction_equality : (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
                            (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3669_366987


namespace NUMINAMATH_CALUDE_monthly_payment_calculation_l3669_366979

def original_price : ℝ := 480
def discount_percentage : ℝ := 5
def first_installment : ℝ := 150
def num_installments : ℕ := 3

theorem monthly_payment_calculation :
  let discounted_price := original_price * (1 - discount_percentage / 100)
  let remaining_balance := discounted_price - first_installment
  let monthly_payment := remaining_balance / num_installments
  monthly_payment = 102 := by
  sorry

end NUMINAMATH_CALUDE_monthly_payment_calculation_l3669_366979


namespace NUMINAMATH_CALUDE_intersection_sum_l3669_366940

/-- Given two lines y = mx + 4 and y = 3x + b intersecting at (6, 10), prove b + m = -7 -/
theorem intersection_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + 4 ↔ y = 3 * x + b) → 
  (6 : ℝ) * m + 4 = 10 → 
  3 * 6 + b = 10 → 
  b + m = -7 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l3669_366940


namespace NUMINAMATH_CALUDE_harolds_marbles_l3669_366921

theorem harolds_marbles (kept : ℕ) (friends : ℕ) (each_friend : ℕ) (initial : ℕ) : 
  kept = 20 → 
  friends = 5 → 
  each_friend = 16 → 
  initial = kept + friends * each_friend → 
  initial = 100 := by
sorry

end NUMINAMATH_CALUDE_harolds_marbles_l3669_366921


namespace NUMINAMATH_CALUDE_system_stable_l3669_366976

-- Define the system of differential equations
def system (x y : ℝ → ℝ) : Prop :=
  ∀ t, (deriv x t = -y t) ∧ (deriv y t = x t)

-- Define Lyapunov stability for the zero solution
def lyapunov_stable (x y : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x₀ y₀ : ℝ,
    x₀^2 + y₀^2 < δ^2 →
    (∀ t ≥ 0, x t^2 + y t^2 < ε^2) ∧
    (x 0 = x₀) ∧ (y 0 = y₀) ∧ system x y

-- Theorem statement
theorem system_stable :
  ∃ x y : ℝ → ℝ, lyapunov_stable x y ∧ system x y ∧ x 0 = 0 ∧ y 0 = 0 :=
sorry

end NUMINAMATH_CALUDE_system_stable_l3669_366976


namespace NUMINAMATH_CALUDE_oil_barrel_ratio_l3669_366934

theorem oil_barrel_ratio (mass_A mass_B : ℝ) : 
  (mass_A + 10000 : ℝ) / (mass_B + 10000) = 4 / 5 →
  (mass_A + 18000 : ℝ) / (mass_B + 2000) = 8 / 7 →
  mass_A / mass_B = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_oil_barrel_ratio_l3669_366934


namespace NUMINAMATH_CALUDE_cookie_sale_charity_share_l3669_366912

/-- Calculates the amount each charity receives when John sells cookies and splits the profit. -/
theorem cookie_sale_charity_share :
  let dozen : ℕ := 6
  let cookies_per_dozen : ℕ := 12
  let total_cookies : ℕ := dozen * cookies_per_dozen
  let price_per_cookie : ℚ := 3/2
  let cost_per_cookie : ℚ := 1/4
  let total_revenue : ℚ := total_cookies * price_per_cookie
  let total_cost : ℚ := total_cookies * cost_per_cookie
  let profit : ℚ := total_revenue - total_cost
  let num_charities : ℕ := 2
  let charity_share : ℚ := profit / num_charities
  charity_share = 45
:= by sorry

end NUMINAMATH_CALUDE_cookie_sale_charity_share_l3669_366912


namespace NUMINAMATH_CALUDE_searchlight_probability_l3669_366956

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℝ := 2

/-- The time in seconds for which the man needs to stay in the dark -/
def dark_time : ℝ := 5

/-- The number of seconds in a minute -/
def seconds_per_minute : ℝ := 60

theorem searchlight_probability :
  let time_per_revolution := seconds_per_minute / revolutions_per_minute
  (dark_time / time_per_revolution : ℝ) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_searchlight_probability_l3669_366956


namespace NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l3669_366971

/-- Given two vectors a and b in ℝ², where a is perpendicular to b,
    prove that the magnitude of their difference is √10. -/
theorem perpendicular_vectors_difference_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : a.1 = x ∧ a.2 = 1)
  (h2 : b = (1, -2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) : 
  ‖(a.1 - b.1, a.2 - b.2)‖ = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l3669_366971


namespace NUMINAMATH_CALUDE_extrema_of_f_l3669_366908

def f (x : ℝ) := -x^2 + x + 1

theorem extrema_of_f :
  let a := 0
  let b := 3/2
  ∃ (x_min x_max : ℝ), x_min ∈ Set.Icc a b ∧ x_max ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    f x_min = 1/4 ∧ f x_max = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_extrema_of_f_l3669_366908


namespace NUMINAMATH_CALUDE_bankers_discount_calculation_l3669_366906

/-- Banker's discount calculation -/
theorem bankers_discount_calculation 
  (present_worth : ℚ) 
  (true_discount : ℚ) 
  (h1 : present_worth = 400)
  (h2 : true_discount = 20) : 
  (true_discount * (present_worth + true_discount)) / present_worth = 21 :=
by
  sorry

#check bankers_discount_calculation

end NUMINAMATH_CALUDE_bankers_discount_calculation_l3669_366906


namespace NUMINAMATH_CALUDE_yushu_donations_l3669_366959

/-- The number of matching combinations for backpacks and pencil cases -/
def matching_combinations (backpack_styles : ℕ) (pencil_case_styles : ℕ) : ℕ :=
  backpack_styles * pencil_case_styles

/-- Theorem: Given 2 backpack styles and 2 pencil case styles, there are 4 matching combinations -/
theorem yushu_donations : matching_combinations 2 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_yushu_donations_l3669_366959


namespace NUMINAMATH_CALUDE_people_left_line_l3669_366961

theorem people_left_line (initial : ℕ) (joined : ℕ) (final : ℕ) (left : ℕ) : 
  initial = 9 → joined = 3 → final = 6 → initial - left + joined = final → left = 6 := by
  sorry

end NUMINAMATH_CALUDE_people_left_line_l3669_366961


namespace NUMINAMATH_CALUDE_oil_bill_ratio_change_l3669_366928

theorem oil_bill_ratio_change 
  (january_bill : ℝ) 
  (february_bill : ℝ) 
  (initial_ratio : ℚ) 
  (added_amount : ℝ) :
  january_bill = 59.99999999999997 →
  initial_ratio = 3 / 2 →
  february_bill / january_bill = initial_ratio →
  (february_bill + added_amount) / january_bill = 5 / 3 →
  added_amount = 10 := by
sorry

end NUMINAMATH_CALUDE_oil_bill_ratio_change_l3669_366928


namespace NUMINAMATH_CALUDE_total_tickets_is_340_l3669_366911

/-- Represents the number of tickets sold for a theater performance. -/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- The total revenue from ticket sales. -/
def totalRevenue (sales : TicketSales) : ℕ :=
  12 * sales.orchestra + 8 * sales.balcony

/-- The difference between balcony and orchestra ticket sales. -/
def balconyOrchestraDiff (sales : TicketSales) : ℤ :=
  sales.balcony - sales.orchestra

/-- The total number of tickets sold. -/
def totalTickets (sales : TicketSales) : ℕ :=
  sales.orchestra + sales.balcony

/-- Theorem stating that given the conditions, the total number of tickets sold is 340. -/
theorem total_tickets_is_340 :
  ∃ (sales : TicketSales),
    totalRevenue sales = 3320 ∧
    balconyOrchestraDiff sales = 40 ∧
    totalTickets sales = 340 := by
  sorry


end NUMINAMATH_CALUDE_total_tickets_is_340_l3669_366911


namespace NUMINAMATH_CALUDE_union_equality_implies_m_value_l3669_366931

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equality_implies_m_value (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_m_value_l3669_366931


namespace NUMINAMATH_CALUDE_first_number_value_l3669_366916

theorem first_number_value (a b c d : ℝ) : 
  (a + b + c) / 3 = 20 →
  (b + c + d) / 3 = 15 →
  d = 18 →
  a = 33 := by
sorry

end NUMINAMATH_CALUDE_first_number_value_l3669_366916


namespace NUMINAMATH_CALUDE_guppies_theorem_l3669_366972

def guppies_problem (haylee_guppies : ℕ) (jose_ratio : ℚ) (charliz_ratio : ℚ) (nicolai_ratio : ℕ) : Prop :=
  let jose_guppies := (haylee_guppies : ℚ) * jose_ratio
  let charliz_guppies := jose_guppies * charliz_ratio
  let nicolai_guppies := (charliz_guppies * nicolai_ratio : ℚ)
  (haylee_guppies : ℚ) + jose_guppies + charliz_guppies + nicolai_guppies = 84

theorem guppies_theorem :
  guppies_problem 36 (1/2) (1/3) 4 :=
sorry

end NUMINAMATH_CALUDE_guppies_theorem_l3669_366972


namespace NUMINAMATH_CALUDE_equation_describes_line_l3669_366992

theorem equation_describes_line :
  ∀ (x y : ℝ), (x - y)^2 = 2*(x^2 + y^2) ↔ y = -x := by sorry

end NUMINAMATH_CALUDE_equation_describes_line_l3669_366992


namespace NUMINAMATH_CALUDE_rain_in_first_hour_l3669_366980

theorem rain_in_first_hour (first_hour : ℝ) (second_hour : ℝ) : 
  second_hour = 2 * first_hour + 7 →
  first_hour + second_hour = 22 →
  first_hour = 5 := by
sorry

end NUMINAMATH_CALUDE_rain_in_first_hour_l3669_366980


namespace NUMINAMATH_CALUDE_prob_two_red_balls_l3669_366999

/-- The probability of selecting 2 red balls from a bag with 5 red, 6 blue, and 4 green balls -/
theorem prob_two_red_balls (red blue green : ℕ) (total : ℕ) (h1 : red = 5) (h2 : blue = 6) (h3 : green = 4) (h4 : total = red + blue + green) :
  (red.choose 2 : ℚ) / (total.choose 2) = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_balls_l3669_366999


namespace NUMINAMATH_CALUDE_total_pencils_count_l3669_366990

/-- The number of pencils each child has -/
def pencils_per_child : ℕ := 2

/-- The number of children -/
def number_of_children : ℕ := 8

/-- The total number of pencils -/
def total_pencils : ℕ := pencils_per_child * number_of_children

theorem total_pencils_count : total_pencils = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_count_l3669_366990


namespace NUMINAMATH_CALUDE_total_boxes_in_cases_l3669_366960

/-- The number of cases Jenny needs to deliver -/
def num_cases : ℕ := 3

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 8

/-- Theorem: The total number of boxes in the cases is 24 -/
theorem total_boxes_in_cases : num_cases * boxes_per_case = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_in_cases_l3669_366960


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3669_366910

theorem geometric_sequence_sum : 
  let a : ℚ := 1/3  -- first term
  let r : ℚ := 1/3  -- common ratio
  let n : ℕ := 5    -- number of terms
  let S : ℚ := (a * (1 - r^n)) / (1 - r)  -- sum formula
  S = 121/243 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3669_366910


namespace NUMINAMATH_CALUDE_least_clock_equivalent_after_six_twelve_is_clock_equivalent_twelve_is_least_clock_equivalent_after_six_l3669_366964

def clock_equivalent (h : ℕ) : Prop :=
  (h ^ 2 - h) % 24 = 0

theorem least_clock_equivalent_after_six :
  ∀ h : ℕ, h > 6 → clock_equivalent h → h ≥ 12 :=
by sorry

theorem twelve_is_clock_equivalent : clock_equivalent 12 :=
by sorry

theorem twelve_is_least_clock_equivalent_after_six :
  ∀ h : ℕ, h > 6 → clock_equivalent h → h = 12 ∨ h > 12 :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_after_six_twelve_is_clock_equivalent_twelve_is_least_clock_equivalent_after_six_l3669_366964


namespace NUMINAMATH_CALUDE_total_wine_age_l3669_366975

-- Define the ages of the wines
def carlo_rosi_age : ℕ := 40
def franzia_age : ℕ := 3 * carlo_rosi_age
def twin_valley_age : ℕ := carlo_rosi_age / 4

-- Theorem statement
theorem total_wine_age :
  franzia_age + carlo_rosi_age + twin_valley_age = 170 :=
by sorry

end NUMINAMATH_CALUDE_total_wine_age_l3669_366975


namespace NUMINAMATH_CALUDE_factorization_equality_l3669_366942

theorem factorization_equality (x y : ℝ) : 
  x^2 - 2*x*y + y^2 - 1 = (x - y + 1) * (x - y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3669_366942


namespace NUMINAMATH_CALUDE_triangle_angle_tangent_l3669_366925

theorem triangle_angle_tangent (A : Real) :
  (Real.sqrt 3 * Real.cos A + Real.sin A) / (Real.sqrt 3 * Real.sin A - Real.cos A) = Real.tan (-7 * π / 12) →
  Real.tan A = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_tangent_l3669_366925


namespace NUMINAMATH_CALUDE_change_for_fifty_cents_l3669_366998

/-- Represents the number of ways to make change for a given amount in cents -/
def makeChange (amount : ℕ) (maxQuarters : ℕ) : ℕ := sorry

/-- The value of a quarter in cents -/
def quarterValue : ℕ := 25

/-- The value of a nickel in cents -/
def nickelValue : ℕ := 5

/-- The value of a penny in cents -/
def pennyValue : ℕ := 1

theorem change_for_fifty_cents :
  makeChange 50 2 = 18 := by sorry

end NUMINAMATH_CALUDE_change_for_fifty_cents_l3669_366998


namespace NUMINAMATH_CALUDE_remainder_of_M_divided_by_500_l3669_366952

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def product_of_factorials : ℕ := (List.range 50).foldl (λ acc i => acc * factorial (i + 1)) 1

def M : ℕ := (product_of_factorials.digits 10).reverse.takeWhile (·= 0) |>.length

theorem remainder_of_M_divided_by_500 : M % 500 = 391 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_M_divided_by_500_l3669_366952


namespace NUMINAMATH_CALUDE_nth_term_is_4021_l3669_366941

/-- An arithmetic sequence with given first three terms -/
structure ArithmeticSequence (x : ℝ) where
  first_term : ℝ := 3 * x - 4
  second_term : ℝ := 6 * x - 17
  third_term : ℝ := 4 * x + 5
  is_arithmetic : second_term - first_term = third_term - second_term

/-- The nth term of the arithmetic sequence -/
def nth_term (seq : ArithmeticSequence x) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * (seq.second_term - seq.first_term)

theorem nth_term_is_4021 (x : ℝ) (seq : ArithmeticSequence x) :
  ∃ n : ℕ, nth_term seq n = 4021 ∧ n = 502 := by
  sorry

end NUMINAMATH_CALUDE_nth_term_is_4021_l3669_366941


namespace NUMINAMATH_CALUDE_seventh_power_equation_l3669_366997

theorem seventh_power_equation (x : ℝ) (hx : x ≠ 0) :
  (7 * x)^5 = (14 * x)^4 ↔ x = 16/7 := by sorry

end NUMINAMATH_CALUDE_seventh_power_equation_l3669_366997


namespace NUMINAMATH_CALUDE_initial_number_proof_l3669_366989

theorem initial_number_proof (x : ℝ) : ((x / 13) / 29) * (1/4) / 2 = 0.125 → x = 754 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l3669_366989


namespace NUMINAMATH_CALUDE_unique_special_parallelogram_l3669_366963

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  B : ℤ × ℤ
  D : ℤ × ℤ
  C : ℚ × ℚ
  area_eq : abs (B.1 * D.2 + D.1 * C.2 + C.1 * 0 - (B.2 * D.1 + D.2 * C.1 + C.2 * 0)) / 2 = 2000000
  B_on_y_eq_x : B.2 = B.1
  D_on_y_eq_2x : D.2 = 2 * D.1
  C_on_y_eq_3x : C.2 = 3 * C.1
  first_quadrant : 0 < B.1 ∧ 0 < B.2 ∧ 0 < D.1 ∧ 0 < D.2 ∧ 0 < C.1 ∧ 0 < C.2
  parallelogram_condition : C.1 = B.1 + D.1 ∧ C.2 = B.2 + D.2

/-- There exists exactly one special parallelogram -/
theorem unique_special_parallelogram : ∃! p : SpecialParallelogram, True :=
  sorry

end NUMINAMATH_CALUDE_unique_special_parallelogram_l3669_366963


namespace NUMINAMATH_CALUDE_profit_increase_l3669_366929

theorem profit_increase (initial_profit : ℝ) (h : initial_profit > 0) :
  let april_profit := initial_profit * 1.2
  let may_profit := april_profit * 0.8
  let june_profit := initial_profit * 1.4399999999999999
  (june_profit / may_profit - 1) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_profit_increase_l3669_366929


namespace NUMINAMATH_CALUDE_second_part_sum_l3669_366927

/-- Calculates the interest on a principal amount for a given rate and time. -/
def interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Proves that given the conditions, the second part of the sum is 1672. -/
theorem second_part_sum (total : ℚ) (first_part : ℚ) (second_part : ℚ) 
  (h1 : total = 2717)
  (h2 : first_part + second_part = total)
  (h3 : interest first_part 3 8 = interest second_part 5 3) :
  second_part = 1672 := by
  sorry

end NUMINAMATH_CALUDE_second_part_sum_l3669_366927


namespace NUMINAMATH_CALUDE_find_number_l3669_366943

theorem find_number (x : ℝ) : x + 1.35 + 0.123 = 1.794 → x = 0.321 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3669_366943


namespace NUMINAMATH_CALUDE_employee_salary_proof_l3669_366909

def total_salary : ℝ := 572
def m_salary_ratio : ℝ := 1.2

theorem employee_salary_proof (n_salary : ℝ) 
  (h1 : n_salary + m_salary_ratio * n_salary = total_salary) :
  n_salary = 260 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_proof_l3669_366909


namespace NUMINAMATH_CALUDE_a_worked_six_days_l3669_366984

/-- Represents the number of days worked by person a -/
def days_a : ℕ := sorry

/-- Represents the daily wage of person a -/
def wage_a : ℕ := sorry

/-- Represents the daily wage of person b -/
def wage_b : ℕ := sorry

/-- Represents the daily wage of person c -/
def wage_c : ℕ := 100

/-- The total earnings of all three workers -/
def total_earnings : ℕ := 1480

/-- Theorem stating that person a worked for 6 days -/
theorem a_worked_six_days :
  (wage_a = 3 * wage_c / 5) ∧
  (wage_b = 4 * wage_c / 5) ∧
  (days_a * wage_a + 9 * wage_b + 4 * wage_c = total_earnings) →
  days_a = 6 :=
by sorry

end NUMINAMATH_CALUDE_a_worked_six_days_l3669_366984


namespace NUMINAMATH_CALUDE_machine_value_depletion_rate_l3669_366958

theorem machine_value_depletion_rate 
  (present_value : ℝ) 
  (value_after_2_years : ℝ) 
  (depletion_rate : ℝ) : 
  present_value = 1100 → 
  value_after_2_years = 891 → 
  value_after_2_years = present_value * (1 - depletion_rate)^2 → 
  depletion_rate = 0.1 := by
sorry

end NUMINAMATH_CALUDE_machine_value_depletion_rate_l3669_366958


namespace NUMINAMATH_CALUDE_women_in_first_group_l3669_366933

/-- The number of women in the first group -/
def first_group : ℕ := 4

/-- The length of cloth colored by the first group in 2 days -/
def cloth_length_first_group : ℕ := 48

/-- The number of days taken by the first group to color the cloth -/
def days_first_group : ℕ := 2

/-- The number of women in the second group -/
def second_group : ℕ := 6

/-- The length of cloth colored by the second group in 1 day -/
def cloth_length_second_group : ℕ := 36

/-- The number of days taken by the second group to color the cloth -/
def days_second_group : ℕ := 1

theorem women_in_first_group : 
  first_group * cloth_length_second_group * days_first_group = 
  second_group * cloth_length_first_group * days_second_group :=
by sorry

end NUMINAMATH_CALUDE_women_in_first_group_l3669_366933


namespace NUMINAMATH_CALUDE_only_one_always_true_l3669_366977

theorem only_one_always_true (a b c : ℝ) : 
  (∃! p : Prop, p = true) ∧ 
  (((a > b → a * c > b * c) = p) ∨
   ((a > b → a^2 * c^2 > b^2 * c^2) = p) ∨
   ((a^2 * c^2 > b^2 * c^2 → a > b) = p)) :=
by sorry

end NUMINAMATH_CALUDE_only_one_always_true_l3669_366977


namespace NUMINAMATH_CALUDE_hash_problem_l3669_366955

-- Define the # operation
def hash (a b : ℚ) : ℚ := a + (a^2 / b)

-- Theorem statement
theorem hash_problem : (hash 4 3) - 10 = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_hash_problem_l3669_366955


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3669_366937

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3669_366937


namespace NUMINAMATH_CALUDE_sequence_properties_l3669_366922

/-- Sequence b_n with sum of first n terms S_n -/
def b : ℕ → ℝ := sorry

/-- Sum of first n terms of b_n -/
def S : ℕ → ℝ := sorry

/-- Arithmetic sequence c_n -/
def c : ℕ → ℝ := sorry

/-- Sequence a_n formed by common terms of b_n and c_n in ascending order -/
def a : ℕ → ℝ := sorry

/-- The product of the first n terms of a_n -/
def T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, 2 * S n = 3 * (b n - 1)) ∧ 
  (c 1 = 5) ∧
  (c 1 + c 2 + c 3 = 27) →
  (∀ n : ℕ, b n = 3^n) ∧
  (∀ n : ℕ, c n = 4*n + 1) ∧
  (T 20 = 9^210) := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3669_366922


namespace NUMINAMATH_CALUDE_silver_division_problem_l3669_366993

/-- 
Given:
- m : ℕ is the number of people
- n : ℕ is the total amount of silver in taels
- Adding 7 taels to each person's share and 7 taels in total equals n
- Subtracting 8 taels from each person's share and subtracting 8 taels in total equals n

Prove that the system of equations 7m + 7 = n and 8m - 8 = n correctly represents the situation
-/
theorem silver_division_problem (m n : ℕ) 
  (h1 : 7 * m + 7 = n) 
  (h2 : 8 * m - 8 = n) : 
  (7 * m + 7 = n) ∧ (8 * m - 8 = n) := by
  sorry

end NUMINAMATH_CALUDE_silver_division_problem_l3669_366993


namespace NUMINAMATH_CALUDE_max_vouchers_for_680_yuan_l3669_366915

/-- Represents the shopping voucher system with a given initial cash amount -/
structure VoucherSystem where
  initial_cash : ℕ
  voucher_rate : ℚ

/-- Calculates the maximum total vouchers that can be received -/
def max_vouchers (system : VoucherSystem) : ℕ :=
  sorry

/-- The theorem stating the maximum vouchers for the given problem -/
theorem max_vouchers_for_680_yuan :
  let system : VoucherSystem := { initial_cash := 680, voucher_rate := 1/5 }
  max_vouchers system = 160 := by
  sorry

end NUMINAMATH_CALUDE_max_vouchers_for_680_yuan_l3669_366915


namespace NUMINAMATH_CALUDE_contractor_absent_days_l3669_366904

theorem contractor_absent_days 
  (total_days : ℕ) 
  (pay_per_day : ℚ) 
  (fine_per_day : ℚ) 
  (total_received : ℚ) 
  (h1 : total_days = 30)
  (h2 : pay_per_day = 25)
  (h3 : fine_per_day = 7.5)
  (h4 : total_received = 425) :
  ∃ (absent_days : ℕ), 
    absent_days = 10 ∧ 
    (total_days - absent_days) * pay_per_day - absent_days * fine_per_day = total_received :=
by sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l3669_366904


namespace NUMINAMATH_CALUDE_min_value_theorem_l3669_366951

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 2) :
  2 / m + 1 / n ≥ 4 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 2 ∧ 2 / m₀ + 1 / n₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3669_366951


namespace NUMINAMATH_CALUDE_octal_subtraction_l3669_366969

/-- Converts a base-8 number represented as a list of digits to a natural number -/
def fromOctal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits -/
def toOctal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: toOctal (n / 8)

/-- The main theorem stating that 5273₈ - 3614₈ = 1457₈ -/
theorem octal_subtraction :
  fromOctal [3, 7, 2, 5] - fromOctal [4, 1, 6, 3] = fromOctal [7, 5, 4, 1] := by
  sorry

#eval toOctal (fromOctal [3, 7, 2, 5] - fromOctal [4, 1, 6, 3])

end NUMINAMATH_CALUDE_octal_subtraction_l3669_366969


namespace NUMINAMATH_CALUDE_infinite_diamond_2005_l3669_366914

/-- A number is diamond 2005 if it has the form ...ab999...99999cd... -/
def is_diamond_2005 (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ) (k m : ℕ), n = a * 10^(k+m+4) + b * 10^(k+m+3) + 999 * 10^m + c * 10 + d

/-- A sequence {a_n} is bounded by C*n if for all n, a_n < C*n -/
def is_bounded_by_linear (a : ℕ → ℕ) (C : ℝ) : Prop :=
  ∀ n, (a n : ℝ) < C * n

/-- A sequence {a_n} is increasing if for all n, a_n <= a_(n+1) -/
def is_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n, a n ≤ a (n + 1)

/-- Main theorem: An increasing sequence bounded by C*n contains infinitely many diamond 2005 numbers -/
theorem infinite_diamond_2005 (a : ℕ → ℕ) (C : ℝ) 
  (h_bound : is_bounded_by_linear a C) 
  (h_incr : is_increasing a) : 
  ∀ m : ℕ, ∃ n > m, is_diamond_2005 (a n) :=
sorry

end NUMINAMATH_CALUDE_infinite_diamond_2005_l3669_366914


namespace NUMINAMATH_CALUDE_cards_in_new_deck_l3669_366966

/-- The number of cards in a new deck -/
def cards_per_deck : ℕ := 55

/-- The number of cards that can be torn at once -/
def cards_per_tear : ℕ := 30

/-- The number of times cards are torn per week -/
def tears_per_week : ℕ := 3

/-- The number of decks purchased -/
def decks_purchased : ℕ := 18

/-- The number of weeks the tearing can continue -/
def weeks_of_tearing : ℕ := 11

/-- Theorem stating the number of cards in a new deck -/
theorem cards_in_new_deck : 
  cards_per_deck * decks_purchased = cards_per_tear * tears_per_week * weeks_of_tearing :=
by sorry

end NUMINAMATH_CALUDE_cards_in_new_deck_l3669_366966


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_30_l3669_366981

/-- An arithmetic sequence and its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with given partial sums, S_30 can be determined -/
theorem arithmetic_sequence_sum_30 (seq : ArithmeticSequence) 
  (h10 : seq.S 10 = 10) (h20 : seq.S 20 = 30) : seq.S 30 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_30_l3669_366981


namespace NUMINAMATH_CALUDE_units_digit_of_2143_power_752_l3669_366930

theorem units_digit_of_2143_power_752 : (2143^752) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2143_power_752_l3669_366930


namespace NUMINAMATH_CALUDE_rectangle_side_greater_than_twelve_l3669_366944

theorem rectangle_side_greater_than_twelve 
  (a b : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : a * b = 3 * (2 * a + 2 * b)) : 
  a > 12 ∨ b > 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_greater_than_twelve_l3669_366944


namespace NUMINAMATH_CALUDE_tree_spacing_l3669_366903

/-- Given 8 equally spaced trees along a straight road, where the distance between
    the first and fifth tree is 100 feet, the distance between the first and last
    tree is 175 feet. -/
theorem tree_spacing (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  (n - 1) * d / 4 = 175 :=
by sorry

end NUMINAMATH_CALUDE_tree_spacing_l3669_366903


namespace NUMINAMATH_CALUDE_expand_product_l3669_366923

theorem expand_product (x : ℝ) : (x + 3) * (x - 4) = x^2 - x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3669_366923


namespace NUMINAMATH_CALUDE_euler_6_years_or_more_percentage_l3669_366917

/-- Represents the number of units for each tenure range in the bar graph --/
structure EmployeeDistribution where
  less_than_2_years : ℕ
  two_to_4_years : ℕ
  four_to_6_years : ℕ
  six_to_8_years : ℕ
  eight_to_10_years : ℕ
  more_than_10_years : ℕ

/-- Calculates the percentage of employees who have worked for 6 years or more --/
def percentage_6_years_or_more (d : EmployeeDistribution) : ℚ :=
  let total := d.less_than_2_years + d.two_to_4_years + d.four_to_6_years +
                d.six_to_8_years + d.eight_to_10_years + d.more_than_10_years
  let six_plus := d.six_to_8_years + d.eight_to_10_years + d.more_than_10_years
  (six_plus : ℚ) / (total : ℚ) * 100

/-- The actual distribution of employees at Euler Company --/
def euler_distribution : EmployeeDistribution :=
  { less_than_2_years := 4
  , two_to_4_years := 6
  , four_to_6_years := 7
  , six_to_8_years := 3
  , eight_to_10_years := 2
  , more_than_10_years := 1 }

theorem euler_6_years_or_more_percentage :
  percentage_6_years_or_more euler_distribution = 26 := by
  sorry

end NUMINAMATH_CALUDE_euler_6_years_or_more_percentage_l3669_366917


namespace NUMINAMATH_CALUDE_problem_solution_l3669_366978

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 7)
  (h_b_a : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3669_366978


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3669_366938

theorem smallest_integer_satisfying_inequality :
  ∃ (y : ℤ), (7 - 3 * y ≤ 29) ∧ (∀ (z : ℤ), z < y → 7 - 3 * z > 29) ∧ y = -7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3669_366938
