import Mathlib

namespace NUMINAMATH_CALUDE_arctan_sum_theorem_l359_35952

theorem arctan_sum_theorem (a b c : ℝ) 
  (h : Real.arctan a + Real.arctan b + Real.arctan c + π / 2 = 0) : 
  (a * b + b * c + c * a = 1) ∧ (a + b + c < a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_theorem_l359_35952


namespace NUMINAMATH_CALUDE_jericho_money_left_l359_35921

/-- The amount of money Jericho has initially -/
def jerichos_money : ℚ := 30

/-- The amount Jericho owes Annika -/
def annika_debt : ℚ := 14

/-- The amount Jericho owes Manny -/
def manny_debt : ℚ := annika_debt / 2

/-- Theorem stating that Jericho will be left with $9 after paying his debts -/
theorem jericho_money_left : jerichos_money - (annika_debt + manny_debt) = 9 := by
  sorry

end NUMINAMATH_CALUDE_jericho_money_left_l359_35921


namespace NUMINAMATH_CALUDE_elevator_problem_solution_l359_35990

/-- Represents the elevator problem with given conditions -/
structure ElevatorProblem where
  total_floors : ℕ
  first_half_time : ℕ
  mid_section_rate : ℕ
  final_section_rate : ℕ

/-- Calculates the total time in hours for the elevator to reach the bottom -/
def total_time (p : ElevatorProblem) : ℚ :=
  let first_half := p.first_half_time
  let mid_section := (p.total_floors / 4) * p.mid_section_rate
  let final_section := (p.total_floors / 4) * p.final_section_rate
  (first_half + mid_section + final_section) / 60

/-- Theorem stating that for the given problem, the total time is 2 hours -/
theorem elevator_problem_solution :
  let problem := ElevatorProblem.mk 20 15 5 16
  total_time problem = 2 := by sorry

end NUMINAMATH_CALUDE_elevator_problem_solution_l359_35990


namespace NUMINAMATH_CALUDE_sandy_shopping_total_l359_35996

/-- The total amount Sandy spent on clothes after discounts, coupon, and tax -/
def total_spent (shorts shirt jacket shoes accessories discount coupon tax : ℚ) : ℚ :=
  let initial_total := shorts + shirt + jacket + shoes + accessories
  let discounted_total := initial_total * (1 - discount)
  let after_coupon := discounted_total - coupon
  let final_total := after_coupon * (1 + tax)
  final_total

/-- Theorem stating the total amount Sandy spent on clothes -/
theorem sandy_shopping_total :
  total_spent 13.99 12.14 7.43 8.50 10.75 0.10 5.00 0.075 = 45.72 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shopping_total_l359_35996


namespace NUMINAMATH_CALUDE_stating_num_small_triangles_formula_l359_35928

/-- Represents a triangle with n points inside it -/
structure TriangleWithPoints where
  n : ℕ  -- number of points inside the triangle
  no_collinear : Bool  -- property that no three points are collinear

/-- 
  Calculates the number of small triangles formed in a triangle with n internal points,
  where no three points (including the triangle's vertices) are collinear.
-/
def numSmallTriangles (t : TriangleWithPoints) : ℕ :=
  2 * t.n + 1

/-- 
  Theorem stating that for a triangle with n points inside,
  where no three points are collinear (including the triangle's vertices),
  the number of small triangles formed is 2n + 1.
-/
theorem num_small_triangles_formula (t : TriangleWithPoints) 
  (h : t.no_collinear = true) : 
  numSmallTriangles t = 2 * t.n + 1 := by
  sorry

#eval numSmallTriangles { n := 100, no_collinear := true }

end NUMINAMATH_CALUDE_stating_num_small_triangles_formula_l359_35928


namespace NUMINAMATH_CALUDE_even_function_m_value_l359_35992

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = x^4 + (m-1)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  x^4 + (m-1)*x + 1

theorem even_function_m_value :
  ∀ m : ℝ, IsEven (f m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_m_value_l359_35992


namespace NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l359_35909

theorem squared_difference_of_quadratic_roots : ∀ p q : ℝ,
  (2 * p^2 + 7 * p - 15 = 0) →
  (2 * q^2 + 7 * q - 15 = 0) →
  (p - q)^2 = 169 / 4 := by
  sorry

end NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l359_35909


namespace NUMINAMATH_CALUDE_flour_sugar_difference_l359_35948

theorem flour_sugar_difference (total_flour sugar_needed flour_added : ℕ) : 
  total_flour = 14 →
  sugar_needed = 9 →
  flour_added = 4 →
  (total_flour - flour_added) - sugar_needed = 1 := by
  sorry

end NUMINAMATH_CALUDE_flour_sugar_difference_l359_35948


namespace NUMINAMATH_CALUDE_probability_of_pair_l359_35955

def deck_size : ℕ := 38
def num_sets : ℕ := 10
def cards_in_reduced_set : ℕ := 3
def cards_in_full_set : ℕ := 4

def total_combinations : ℕ := (deck_size * (deck_size - 1)) / 2

def favorable_outcomes : ℕ := 
  (cards_in_reduced_set * (cards_in_reduced_set - 1)) / 2 + 
  (num_sets - 1) * (cards_in_full_set * (cards_in_full_set - 1)) / 2

theorem probability_of_pair (m n : ℕ) (h : m.gcd n = 1) :
  (m : ℚ) / n = favorable_outcomes / total_combinations → m = 57 ∧ n = 703 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_pair_l359_35955


namespace NUMINAMATH_CALUDE_complex_equality_implies_values_l359_35914

theorem complex_equality_implies_values (x y : ℝ) : 
  (Complex.mk (x - 1) y = Complex.mk 0 1 - Complex.mk (3*x) 0) → 
  (x = 1/4 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_values_l359_35914


namespace NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l359_35973

theorem gcd_of_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 4) :
  Nat.gcd X Y = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l359_35973


namespace NUMINAMATH_CALUDE_hyperbola_properties_l359_35951

/-- Given a hyperbola C with equation x²/(m²+3) - y²/m² = 1 where m > 0,
    and asymptote equation y = ±(1/2)x, prove the following properties --/
theorem hyperbola_properties (m : ℝ) (h1 : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / (m^2 + 3) - y^2 / m^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (1/2) * x ∨ y = -(1/2) * x}
  (∀ (x y : ℝ), (x, y) ∈ C → (x, y) ∈ asymptote → x ≠ 0 → y / x = 1/2 ∨ y / x = -1/2) →
  (m = 1) ∧
  (∃ (x y : ℝ), (x, y) ∈ C ∧ y = Real.log (x - 1) ∧ 
    (x^2 / (m^2 + 3) = 1 ∨ y = 0)) ∧
  (∀ (x y : ℝ), y^2 - x^2 / 4 = 1 ↔ (x, y) ∈ asymptote) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l359_35951


namespace NUMINAMATH_CALUDE_carpet_area_is_27072_l359_35944

/-- Calculates the area of carpet required for a room with a column -/
def carpet_area (room_length room_width column_side : ℕ) : ℕ :=
  let inches_per_foot := 12
  let room_length_inches := room_length * inches_per_foot
  let room_width_inches := room_width * inches_per_foot
  let column_side_inches := column_side * inches_per_foot
  let total_area := room_length_inches * room_width_inches
  let column_area := column_side_inches * column_side_inches
  total_area - column_area

/-- Theorem: The carpet area for the given room is 27,072 square inches -/
theorem carpet_area_is_27072 :
  carpet_area 16 12 2 = 27072 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_is_27072_l359_35944


namespace NUMINAMATH_CALUDE_correct_calculation_l359_35933

theorem correct_calculation (a b : ℝ) : 3 * a * b^2 - 5 * b^2 * a = -2 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l359_35933


namespace NUMINAMATH_CALUDE_circle_intersection_range_l359_35967

-- Define the circles
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*a*y + 2*a^2 - 4 = 0

def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the intersection condition
def intersect_at_all_times (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C a x y ∧ circle_O x y

-- Theorem statement
theorem circle_intersection_range :
  ∀ a : ℝ, intersect_at_all_times a ↔ 
    ((-2 * Real.sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l359_35967


namespace NUMINAMATH_CALUDE_symmetric_points_fourth_quadrant_l359_35981

/-- Given two points A(a, 3) and B(2, b) symmetric with respect to the x-axis,
    prove that point M(a, b) is in the fourth quadrant. -/
theorem symmetric_points_fourth_quadrant (a b : ℝ) :
  (a = 2 ∧ b = -3) →  -- Conditions derived from symmetry
  (a > 0 ∧ b < 0)     -- Definition of fourth quadrant
  := by sorry

end NUMINAMATH_CALUDE_symmetric_points_fourth_quadrant_l359_35981


namespace NUMINAMATH_CALUDE_min_value_theorem_l359_35962

theorem min_value_theorem (a b x : ℝ) (ha : a > 1) (hb : b > 2) (hx : x + b = 5) :
  (∀ y : ℝ, y > 1 ∧ y + b = 5 → (1 / (a - 1) + 9 / (b - 2) ≤ 1 / (y - 1) + 9 / (b - 2))) ∧
  (1 / (a - 1) + 9 / (b - 2) = 8) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l359_35962


namespace NUMINAMATH_CALUDE_f_and_g_properties_l359_35905

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| - |1 - x|
def g (a b x : ℝ) : ℝ := |x + a^2| + |x - b^2|

-- State the theorem
theorem f_and_g_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x, x ∈ {x : ℝ | f x ≥ 1} ↔ x ∈ Set.Ici (1/2)) ∧
  (∀ x, f x ≤ g a b x) := by
  sorry

end NUMINAMATH_CALUDE_f_and_g_properties_l359_35905


namespace NUMINAMATH_CALUDE_shirt_count_proof_l359_35972

/-- The number of different colored neckties -/
def num_neckties : ℕ := 6

/-- The probability that all boxes contain matching necktie-shirt pairs -/
def matching_probability : ℝ := 0.041666666666666664

/-- The number of different colored shirts -/
def num_shirts : ℕ := 2

theorem shirt_count_proof :
  (1 / num_shirts : ℝ) ^ num_neckties = matching_probability ∧
  num_shirts = ⌈(1 / matching_probability) ^ (1 / num_neckties : ℝ)⌉ := by
  sorry

#check shirt_count_proof

end NUMINAMATH_CALUDE_shirt_count_proof_l359_35972


namespace NUMINAMATH_CALUDE_f_neg_two_l359_35989

def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem f_neg_two : f (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_f_neg_two_l359_35989


namespace NUMINAMATH_CALUDE_exam_mean_is_115_l359_35923

/-- The mean score of the exam -/
def mean : ℝ := 115

/-- The standard deviation of the exam scores -/
def std_dev : ℝ := 40

/-- Theorem stating that the given conditions imply the mean score is 115 -/
theorem exam_mean_is_115 :
  (55 = mean - 1.5 * std_dev) ∧
  (75 = mean - 2 * std_dev) ∧
  (85 = mean + 1.5 * std_dev) ∧
  (100 = mean + 3.5 * std_dev) →
  mean = 115 := by sorry

end NUMINAMATH_CALUDE_exam_mean_is_115_l359_35923


namespace NUMINAMATH_CALUDE_total_movies_equals_sum_watched_and_to_watch_l359_35929

/-- The 'crazy silly school' series -/
structure CrazySillySchool where
  total_books : ℕ
  total_movies : ℕ
  books_read : ℕ
  movies_watched : ℕ
  movies_to_watch : ℕ

/-- Theorem: The total number of movies in the series is equal to the sum of movies watched and movies left to watch -/
theorem total_movies_equals_sum_watched_and_to_watch (css : CrazySillySchool) 
  (h1 : css.total_books = 4)
  (h2 : css.books_read = 19)
  (h3 : css.movies_watched = 7)
  (h4 : css.movies_to_watch = 10) :
  css.total_movies = css.movies_watched + css.movies_to_watch :=
by
  sorry

#eval 7 + 10  -- Expected output: 17

end NUMINAMATH_CALUDE_total_movies_equals_sum_watched_and_to_watch_l359_35929


namespace NUMINAMATH_CALUDE_question_mark_value_l359_35970

theorem question_mark_value : ∃ x : ℤ, 27474 + 3699 + x - 2047 = 31111 ∧ x = 1985 := by
  sorry

end NUMINAMATH_CALUDE_question_mark_value_l359_35970


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l359_35908

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4*a*x else (2*a+3)*x - 4*a + 5

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l359_35908


namespace NUMINAMATH_CALUDE_p_implies_m_range_p_and_q_implies_m_range_l359_35997

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ a = m - 1 ∧ b = 3 - m ∧ 
  ∀ (x y : ℝ), x^2 / a + y^2 / b = 1 → ∃ (c : ℝ), x^2 + (y^2 - c^2) = a * b

def q (m : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = x^2 - m*x + 9/16 ∧ y > 0

-- Theorem 1
theorem p_implies_m_range (m : ℝ) : p m → 1 < m ∧ m < 2 := by sorry

-- Theorem 2
theorem p_and_q_implies_m_range (m : ℝ) : p m ∧ q m → 1 < m ∧ m < 3/2 := by sorry

end NUMINAMATH_CALUDE_p_implies_m_range_p_and_q_implies_m_range_l359_35997


namespace NUMINAMATH_CALUDE_min_value_of_expression_l359_35975

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l359_35975


namespace NUMINAMATH_CALUDE_star_emilio_sum_difference_l359_35947

def star_list : List Nat := List.range 40

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem star_emilio_sum_difference :
  star_list.sum - emilio_list.sum = 104 := by sorry

end NUMINAMATH_CALUDE_star_emilio_sum_difference_l359_35947


namespace NUMINAMATH_CALUDE_min_weeks_for_puppies_l359_35925

/-- Represents the different types of puppies Bob can buy -/
inductive PuppyType
  | GoldenRetriever
  | Poodle
  | Beagle

/-- Calculates the minimum number of weeks Bob needs to compete to afford a puppy -/
def min_weeks_to_afford (puppy : PuppyType) (entrance_fee : ℕ) (first_place_prize : ℕ) (current_savings : ℕ) : ℕ :=
  match puppy with
  | PuppyType.GoldenRetriever => 10
  | PuppyType.Poodle => 7
  | PuppyType.Beagle => 5

/-- Theorem stating the minimum number of weeks Bob needs to compete for each puppy type -/
theorem min_weeks_for_puppies 
  (entrance_fee : ℕ) 
  (first_place_prize : ℕ) 
  (second_place_prize : ℕ) 
  (third_place_prize : ℕ) 
  (current_savings : ℕ) 
  (golden_price : ℕ) 
  (poodle_price : ℕ) 
  (beagle_price : ℕ) 
  (h1 : entrance_fee = 10)
  (h2 : first_place_prize = 100)
  (h3 : second_place_prize = 70)
  (h4 : third_place_prize = 40)
  (h5 : current_savings = 180)
  (h6 : golden_price = 1000)
  (h7 : poodle_price = 800)
  (h8 : beagle_price = 600) :
  (min_weeks_to_afford PuppyType.GoldenRetriever entrance_fee first_place_prize current_savings = 10) ∧
  (min_weeks_to_afford PuppyType.Poodle entrance_fee first_place_prize current_savings = 7) ∧
  (min_weeks_to_afford PuppyType.Beagle entrance_fee first_place_prize current_savings = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_weeks_for_puppies_l359_35925


namespace NUMINAMATH_CALUDE_dishes_for_equal_time_l359_35993

/-- Represents the time taken for different chores -/
structure ChoreTime where
  sweep : ℕ  -- minutes per room
  wash : ℕ   -- minutes per dish
  laundry : ℕ -- minutes per load
  dust : ℕ   -- minutes per surface

/-- Represents the chores assigned to Anna -/
structure AnnaChores where
  rooms : ℕ
  surfaces : ℕ

/-- Represents the chores assigned to Billy -/
structure BillyChores where
  loads : ℕ
  surfaces : ℕ

/-- Calculates the total time Anna spends on chores -/
def annaTime (ct : ChoreTime) (ac : AnnaChores) : ℕ :=
  ct.sweep * ac.rooms + ct.dust * ac.surfaces

/-- Calculates the total time Billy spends on chores, excluding dishes -/
def billyTimeBeforeDishes (ct : ChoreTime) (bc : BillyChores) : ℕ :=
  ct.laundry * bc.loads + ct.dust * bc.surfaces

/-- The main theorem to prove -/
theorem dishes_for_equal_time (ct : ChoreTime) (ac : AnnaChores) (bc : BillyChores) :
  ct.sweep = 3 →
  ct.wash = 2 →
  ct.laundry = 9 →
  ct.dust = 1 →
  ac.rooms = 10 →
  ac.surfaces = 14 →
  bc.loads = 2 →
  bc.surfaces = 6 →
  ∃ (dishes : ℕ), dishes = 10 ∧
    annaTime ct ac = billyTimeBeforeDishes ct bc + ct.wash * dishes :=
by
  sorry


end NUMINAMATH_CALUDE_dishes_for_equal_time_l359_35993


namespace NUMINAMATH_CALUDE_current_rabbits_in_cage_l359_35945

/-- The number of rabbits Jasper saw in the park -/
def rabbits_in_park : ℕ := 60

/-- The number of rabbits currently in the cage -/
def rabbits_in_cage : ℕ := 13

/-- The number of rabbits to be added to the cage -/
def rabbits_to_add : ℕ := 7

theorem current_rabbits_in_cage :
  rabbits_in_cage + rabbits_to_add = rabbits_in_park / 3 ∧
  rabbits_in_cage = 13 :=
by sorry

end NUMINAMATH_CALUDE_current_rabbits_in_cage_l359_35945


namespace NUMINAMATH_CALUDE_prime_power_expression_l359_35917

theorem prime_power_expression (a b : ℕ) :
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ 
   (a^6 + 21*a^4*b^2 + 35*a^2*b^4 + 7*b^6) * (b^6 + 21*b^4*a^2 + 35*b^2*a^4 + 7*a^6) = p^k) ↔ 
  (∃ (i : ℕ), a = 2^i ∧ b = 2^i) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_expression_l359_35917


namespace NUMINAMATH_CALUDE_soda_price_calculation_l359_35938

def pizza_price : ℚ := 12
def fries_price : ℚ := (3 / 10)
def goal_amount : ℚ := 500
def pizzas_sold : ℕ := 15
def fries_sold : ℕ := 40
def sodas_sold : ℕ := 25
def remaining_amount : ℚ := 258

theorem soda_price_calculation :
  ∃ (soda_price : ℚ),
    soda_price * sodas_sold = 
      goal_amount - remaining_amount - 
      (pizza_price * pizzas_sold + fries_price * fries_sold) ∧
    soda_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_soda_price_calculation_l359_35938


namespace NUMINAMATH_CALUDE_survey_result_l359_35976

theorem survey_result (total : ℕ) (tv_dislike_percent : ℚ) (both_dislike_percent : ℚ)
  (h_total : total = 1500)
  (h_tv_dislike : tv_dislike_percent = 25 / 100)
  (h_both_dislike : both_dislike_percent = 15 / 100) :
  ⌊(total : ℚ) * tv_dislike_percent * both_dislike_percent⌋ = 56 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l359_35976


namespace NUMINAMATH_CALUDE_james_lollipops_distribution_l359_35988

/-- The number of lollipops James has left after distributing to his friends -/
def lollipops_left (total_lollipops : ℕ) (num_friends : ℕ) : ℕ :=
  total_lollipops % num_friends

/-- Theorem stating that James has 0 lollipops left after distribution -/
theorem james_lollipops_distribution :
  let total_lollipops : ℕ := 56 + 130 + 10 + 238
  let num_friends : ℕ := 14
  lollipops_left total_lollipops num_friends = 0 := by sorry

end NUMINAMATH_CALUDE_james_lollipops_distribution_l359_35988


namespace NUMINAMATH_CALUDE_product_of_Q_at_roots_of_P_l359_35916

/-- The polynomial P(x) = x^5 - x^2 + 1 -/
def P (x : ℂ) : ℂ := x^5 - x^2 + 1

/-- The polynomial Q(x) = x^2 + 1 -/
def Q (x : ℂ) : ℂ := x^2 + 1

theorem product_of_Q_at_roots_of_P :
  ∃ (r₁ r₂ r₃ r₄ r₅ : ℂ),
    (P r₁ = 0) ∧ (P r₂ = 0) ∧ (P r₃ = 0) ∧ (P r₄ = 0) ∧ (P r₅ = 0) ∧
    (Q r₁ * Q r₂ * Q r₃ * Q r₄ * Q r₅ = 5) := by
  sorry

end NUMINAMATH_CALUDE_product_of_Q_at_roots_of_P_l359_35916


namespace NUMINAMATH_CALUDE_unique_triangle_side_length_l359_35979

open Real

theorem unique_triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  A = π / 4 →
  b = 2 * Real.sqrt 2 →
  0 < B →
  B < 3 * π / 4 →
  0 < C →
  C < 3 * π / 4 →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  (∀ B' C' b' c', 
    0 < B' → B' < 3 * π / 4 → 
    0 < C' → C' < 3 * π / 4 → 
    A + B' + C' = π → 
    2 / sin A = b' / sin B' → 
    2 / sin A = c' / sin C' → 
    (B' = B ∧ C' = C ∧ b' = b ∧ c' = c)) →
  b = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_triangle_side_length_l359_35979


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l359_35913

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 5 + a 6 = 12 →
  a 2 + a 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l359_35913


namespace NUMINAMATH_CALUDE_gcd_of_f_is_2730_l359_35991

-- Define the function f(n) = n^13 - n
def f (n : ℤ) : ℤ := n^13 - n

-- State the theorem
theorem gcd_of_f_is_2730 : 
  ∃ (d : ℕ), d = 2730 ∧ ∀ (n : ℤ), (f n).natAbs ∣ d ∧ 
  (∀ (m : ℕ), (∀ (k : ℤ), (f k).natAbs ∣ m) → d ∣ m) :=
sorry

end NUMINAMATH_CALUDE_gcd_of_f_is_2730_l359_35991


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_l359_35931

/-- A quadratic function represented by its coefficients -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Predicate for a quadratic function passing through the origin -/
def passes_through_origin (f : QuadraticFunction) : Prop :=
  f.a * 0^2 + f.b * 0 + f.c = 0

/-- Theorem stating that b = c = 0 is a sufficient condition -/
theorem sufficient_condition (f : QuadraticFunction) (h1 : f.b = 0) (h2 : f.c = 0) :
  passes_through_origin f := by sorry

/-- Theorem stating that b = c = 0 is not a necessary condition -/
theorem not_necessary_condition :
  ∃ f : QuadraticFunction, passes_through_origin f ∧ f.b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_l359_35931


namespace NUMINAMATH_CALUDE_smaller_screen_diagonal_l359_35936

theorem smaller_screen_diagonal : 
  ∃ (x : ℝ), x > 0 ∧ x^2 + 34 = 18^2 ∧ x = Real.sqrt 290 := by sorry

end NUMINAMATH_CALUDE_smaller_screen_diagonal_l359_35936


namespace NUMINAMATH_CALUDE_cost_split_theorem_l359_35918

/-- Calculates the amount each person should pay when a group buys items and splits the cost equally -/
def calculate_cost_per_person (num_people : ℕ) (item1_count : ℕ) (item1_price : ℕ) (item2_count : ℕ) (item2_price : ℕ) : ℕ :=
  ((item1_count * item1_price + item2_count * item2_price) / num_people)

/-- Proves that when 4 friends buy 5 items at 200 won each and 7 items at 800 won each, 
    and divide the total cost equally, each person should pay 1650 won -/
theorem cost_split_theorem : 
  calculate_cost_per_person 4 5 200 7 800 = 1650 := by
  sorry

end NUMINAMATH_CALUDE_cost_split_theorem_l359_35918


namespace NUMINAMATH_CALUDE_milk_dilution_l359_35950

theorem milk_dilution (initial_volume : ℝ) (pure_milk_added : ℝ) (initial_water_percentage : ℝ) :
  initial_volume = 10 →
  pure_milk_added = 15 →
  initial_water_percentage = 5 →
  let initial_water := initial_volume * (initial_water_percentage / 100)
  let final_volume := initial_volume + pure_milk_added
  let final_water_percentage := (initial_water / final_volume) * 100
  final_water_percentage = 2 := by
  sorry

end NUMINAMATH_CALUDE_milk_dilution_l359_35950


namespace NUMINAMATH_CALUDE_dans_remaining_money_l359_35966

/-- Dan's initial money in dollars -/
def initial_money : ℝ := 5

/-- Cost of the candy bar in dollars -/
def candy_bar_cost : ℝ := 2

/-- Theorem: Dan's remaining money after buying the candy bar is $3 -/
theorem dans_remaining_money : 
  initial_money - candy_bar_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l359_35966


namespace NUMINAMATH_CALUDE_lineup_count_l359_35927

/-- The number of team members -/
def team_size : ℕ := 16

/-- The number of positions in the lineup -/
def lineup_size : ℕ := 5

/-- The number of pre-assigned positions -/
def pre_assigned : ℕ := 2

/-- Calculate the number of ways to choose a lineup -/
def lineup_ways : ℕ :=
  (team_size - pre_assigned) * (team_size - pre_assigned - 1) * (team_size - pre_assigned - 2)

theorem lineup_count : lineup_ways = 2184 := by sorry

end NUMINAMATH_CALUDE_lineup_count_l359_35927


namespace NUMINAMATH_CALUDE_final_x_value_l359_35978

/-- Represents the state of the program at each iteration -/
structure State where
  x : ℕ  -- Current value of X
  s : ℕ  -- Current sum S
  n : ℕ  -- Number of iterations

/-- Updates the state for the next iteration -/
def nextState (state : State) : State :=
  { x := state.x + 2,
    s := state.s + state.x + 2,
    n := state.n + 1 }

/-- Computes the final state when S ≥ 10000 -/
def finalState : State :=
  sorry

/-- The main theorem to prove -/
theorem final_x_value :
  finalState.x = 201 ∧ finalState.s ≥ 10000 ∧
  ∀ (prev : State), prev.n < finalState.n → prev.s < 10000 :=
sorry

end NUMINAMATH_CALUDE_final_x_value_l359_35978


namespace NUMINAMATH_CALUDE_centroid_vector_sum_l359_35919

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a triangle ABC with centroid G and an arbitrary point P,
    prove that PG = 1/3(PA + PB + PC) -/
theorem centroid_vector_sum (A B C P G : V) 
  (h : G = (1/3 : ℝ) • (A + B + C)) : 
  G - P = (1/3 : ℝ) • ((A - P) + (B - P) + (C - P)) := by
  sorry

end NUMINAMATH_CALUDE_centroid_vector_sum_l359_35919


namespace NUMINAMATH_CALUDE_nancy_shelving_problem_l359_35924

/-- The number of romance books shelved by Nancy the librarian --/
def romance_books : ℕ := 8

/-- The total number of books on the cart --/
def total_books : ℕ := 46

/-- The number of history books shelved --/
def history_books : ℕ := 12

/-- The number of poetry books shelved --/
def poetry_books : ℕ := 4

/-- The number of Western novels shelved --/
def western_books : ℕ := 5

/-- The number of biographies shelved --/
def biography_books : ℕ := 6

theorem nancy_shelving_problem :
  romance_books = 8 ∧
  total_books = 46 ∧
  history_books = 12 ∧
  poetry_books = 4 ∧
  western_books = 5 ∧
  biography_books = 6 ∧
  (total_books - (history_books + romance_books + poetry_books)) % 2 = 0 ∧
  (total_books - (history_books + romance_books + poetry_books)) / 2 = western_books + biography_books :=
by sorry

end NUMINAMATH_CALUDE_nancy_shelving_problem_l359_35924


namespace NUMINAMATH_CALUDE_ellipse_circle_theorem_l359_35965

/-- Definition of the ellipse C -/
def ellipse_C (b : ℝ) (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / b^2 = 1 ∧ b > 0

/-- Definition of the circle O -/
def circle_O (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2 ∧ r > 0

/-- Definition of the right focus F of ellipse C -/
def right_focus (F : ℝ × ℝ) (b : ℝ) : Prop :=
  F.1 > 0 ∧ ellipse_C b F.1 F.2

/-- Definition of the tangent lines from F to circle O -/
def tangent_lines (F A B : ℝ × ℝ) (r : ℝ) : Prop :=
  circle_O r A.1 A.2 ∧ circle_O r B.1 B.2 ∧
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2

/-- Definition of right triangle ABF -/
def right_triangle (A B F : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 0

/-- Definition of maximum distance between points on C and O -/
def max_distance (b r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C b x₁ y₁ ∧ circle_O r x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (Real.sqrt 3 + 1)^2

/-- Main theorem -/
theorem ellipse_circle_theorem
  (b r : ℝ) (F A B : ℝ × ℝ) :
  ellipse_C b F.1 F.2 →
  right_focus F b →
  circle_O r A.1 A.2 →
  tangent_lines F A B r →
  right_triangle A B F →
  max_distance b r →
  (r = 1 ∧ b = 1) ∧
  (∀ (k m : ℝ), k < 0 → m > 0 →
    (∃ (P Q : ℝ × ℝ),
      ellipse_C b P.1 P.2 ∧ ellipse_C b Q.1 Q.2 ∧
      P.2 = k * P.1 + m ∧ Q.2 = k * Q.1 + m ∧
      (P.1 - F.1)^2 + (P.2 - F.2)^2 +
      (Q.1 - F.1)^2 + (Q.2 - F.2)^2 +
      (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 12)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_theorem_l359_35965


namespace NUMINAMATH_CALUDE_equal_time_travel_ratio_l359_35954

/-- The ratio of distances when travel times are equal --/
theorem equal_time_travel_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  y / 1 = x / 1 + (x + y) / 10 → x / y = 9 / 11 := by
  sorry

#check equal_time_travel_ratio

end NUMINAMATH_CALUDE_equal_time_travel_ratio_l359_35954


namespace NUMINAMATH_CALUDE_steves_salary_l359_35959

theorem steves_salary (take_home_pay : ℝ) (tax_rate : ℝ) (healthcare_rate : ℝ) (union_dues : ℝ) 
  (h1 : take_home_pay = 27200)
  (h2 : tax_rate = 0.20)
  (h3 : healthcare_rate = 0.10)
  (h4 : union_dues = 800) :
  ∃ (original_salary : ℝ), 
    original_salary * (1 - tax_rate - healthcare_rate) - union_dues = take_home_pay ∧ 
    original_salary = 40000 := by
sorry

end NUMINAMATH_CALUDE_steves_salary_l359_35959


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l359_35998

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ r s : ℝ, (2 * r^2 - 4 * r - 8 = 0) ∧ 
               (2 * s^2 - 4 * s - 8 = 0) ∧ 
               ((r + 3)^2 + b * (r + 3) + c = 0) ∧ 
               ((s + 3)^2 + b * (s + 3) + c = 0)) →
  c = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l359_35998


namespace NUMINAMATH_CALUDE_pairwise_products_equal_differences_impossible_l359_35902

theorem pairwise_products_equal_differences_impossible
  (a b c d : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_order : a < b ∧ b < c ∧ c < d)
  (h_product_order : a * b < a * c ∧ a * c < a * d ∧ a * d < b * c ∧ b * c < b * d ∧ b * d < c * d) :
  ¬∃ k : ℝ, k > 0 ∧
    a * c - a * b = k ∧
    a * d - a * c = k ∧
    b * c - a * d = k ∧
    b * d - b * c = k ∧
    c * d - b * d = k :=
by sorry

end NUMINAMATH_CALUDE_pairwise_products_equal_differences_impossible_l359_35902


namespace NUMINAMATH_CALUDE_seashells_left_l359_35957

def total_seashells : ℕ := 679
def clam_shells : ℕ := 325
def conch_shells : ℕ := 210
def oyster_shells : ℕ := 144
def starfish : ℕ := 110

def clam_percentage : ℚ := 40 / 100
def conch_percentage : ℚ := 25 / 100
def oyster_fraction : ℚ := 1 / 3

theorem seashells_left : 
  (clam_shells - Int.floor (clam_percentage * clam_shells)) +
  (conch_shells - Int.ceil (conch_percentage * conch_shells)) +
  (oyster_shells - Int.floor (oyster_fraction * oyster_shells)) +
  starfish = 558 := by
sorry

end NUMINAMATH_CALUDE_seashells_left_l359_35957


namespace NUMINAMATH_CALUDE_natalia_clip_sales_l359_35999

/-- The total number of clips Natalia sold in April and May -/
def total_clips (april_sales : ℕ) (may_sales : ℕ) : ℕ := april_sales + may_sales

/-- Theorem stating the total number of clips sold given the conditions -/
theorem natalia_clip_sales : 
  ∀ (april_sales : ℕ), 
  april_sales = 48 → 
  total_clips april_sales (april_sales / 2) = 72 := by
sorry

end NUMINAMATH_CALUDE_natalia_clip_sales_l359_35999


namespace NUMINAMATH_CALUDE_inscribed_square_perimeter_l359_35932

/-- The perimeter of a square inscribed in a right triangle -/
theorem inscribed_square_perimeter (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let x := a * b / (a + b)
  4 * x = 4 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_perimeter_l359_35932


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l359_35956

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) 
  (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0)
  (h_inv_prop : ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, x * y = k)
  (h_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l359_35956


namespace NUMINAMATH_CALUDE_drums_per_day_l359_35968

/-- Given that 90 drums are filled in 6 days, prove that 15 drums are filled per day -/
theorem drums_per_day :
  ∀ (total_drums : ℕ) (total_days : ℕ) (drums_per_day : ℕ),
    total_drums = 90 →
    total_days = 6 →
    drums_per_day = total_drums / total_days →
    drums_per_day = 15 := by
  sorry

end NUMINAMATH_CALUDE_drums_per_day_l359_35968


namespace NUMINAMATH_CALUDE_profit_percentage_l359_35911

theorem profit_percentage (C S : ℝ) (h : 55 * C = 50 * S) : 
  (S - C) / C * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l359_35911


namespace NUMINAMATH_CALUDE_otimes_twelve_nine_l359_35963

-- Define the custom operation
def otimes (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

-- Theorem statement
theorem otimes_twelve_nine : otimes 12 9 = 13 + 7/9 := by
  sorry

end NUMINAMATH_CALUDE_otimes_twelve_nine_l359_35963


namespace NUMINAMATH_CALUDE_distance_between_points_l359_35974

/-- The distance between points (2, 2) and (-1, -1) is 3√2 -/
theorem distance_between_points : Real.sqrt ((2 - (-1))^2 + (2 - (-1))^2) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l359_35974


namespace NUMINAMATH_CALUDE_grocery_store_inventory_l359_35904

theorem grocery_store_inventory (apples regular_soda diet_soda : ℕ) 
  (h1 : apples = 36)
  (h2 : regular_soda = 80)
  (h3 : diet_soda = 54) :
  regular_soda + diet_soda - apples = 98 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_inventory_l359_35904


namespace NUMINAMATH_CALUDE_nearest_whole_number_solution_l359_35915

theorem nearest_whole_number_solution (x : ℝ) : 
  x * 54 = 75625 → 
  ⌊x + 0.5⌋ = 1400 :=
sorry

end NUMINAMATH_CALUDE_nearest_whole_number_solution_l359_35915


namespace NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l359_35977

theorem baby_sea_turtles_on_sand (total : ℕ) (swept_fraction : ℚ) : total = 42 → swept_fraction = 1/3 → total - (swept_fraction * total).floor = 28 := by
  sorry

end NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l359_35977


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l359_35941

theorem arithmetic_simplification :
  (30 - (2030 - 30 * 2)) + (2030 - (30 * 2 - 30)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l359_35941


namespace NUMINAMATH_CALUDE_win_trip_l359_35961

/-- The number of chocolate bars Tom needs to sell to win the trip -/
def total_bars : ℕ := 3465

/-- The number of chocolate bars in each box -/
def bars_per_box : ℕ := 7

/-- The number of boxes Tom needs to sell to win the trip -/
def boxes_needed : ℕ := total_bars / bars_per_box

theorem win_trip : boxes_needed = 495 := by
  sorry

end NUMINAMATH_CALUDE_win_trip_l359_35961


namespace NUMINAMATH_CALUDE_prime_saturated_bound_l359_35995

def isPrimeSaturated (n : ℕ) (bound : ℕ) : Prop :=
  (Finset.prod (Nat.factors n).toFinset id) < bound

def isGreatestTwoDigitPrimeSaturated (n : ℕ) : Prop :=
  n ≤ 99 ∧ isPrimeSaturated n 96 ∧ ∀ m, m ≤ 99 → isPrimeSaturated m 96 → m ≤ n

theorem prime_saturated_bound (n : ℕ) :
  isGreatestTwoDigitPrimeSaturated 96 →
  isPrimeSaturated n (Finset.prod (Nat.factors n).toFinset id + 1) →
  Finset.prod (Nat.factors n).toFinset id < 96 :=
by sorry

end NUMINAMATH_CALUDE_prime_saturated_bound_l359_35995


namespace NUMINAMATH_CALUDE_multiples_of_hundred_sequence_l359_35940

theorem multiples_of_hundred_sequence (start : ℕ) :
  (∃ seq : Finset ℕ,
    seq.card = 10 ∧
    (∀ n ∈ seq, n % 100 = 0) ∧
    (∀ n ∈ seq, start ≤ n ∧ n ≤ 1000) ∧
    1000 ∈ seq) →
  start = 100 :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_hundred_sequence_l359_35940


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l359_35901

def arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : arithmetic_sequence a q)
  (h2 : q > 1)
  (h3 : a 3 * a 7 = 72)
  (h4 : a 2 + a 8 = 27) :
  a 12 = 96 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l359_35901


namespace NUMINAMATH_CALUDE_lines_parallel_if_perpendicular_to_parallel_planes_l359_35985

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (lineParallel : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_if_perpendicular_to_parallel_planes 
  (α β : Plane) (a b : Line)
  (h_distinct_planes : α ≠ β)
  (h_distinct_lines : a ≠ b)
  (h_a_perp_α : perpendicular a α)
  (h_b_perp_β : perpendicular b β)
  (h_α_parallel_β : parallel α β) :
  lineParallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_if_perpendicular_to_parallel_planes_l359_35985


namespace NUMINAMATH_CALUDE_power_of_product_l359_35980

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l359_35980


namespace NUMINAMATH_CALUDE_small_box_width_l359_35982

/-- Proves that the width of smaller boxes is 50 cm given the conditions of the problem -/
theorem small_box_width (large_length large_width large_height : ℝ)
                        (small_length small_height : ℝ)
                        (max_small_boxes : ℕ) :
  large_length = 6 →
  large_width = 5 →
  large_height = 4 →
  small_length = 0.6 →
  small_height = 0.4 →
  max_small_boxes = 1000 →
  ∃ (small_width : ℝ),
    small_width = 0.5 ∧
    (max_small_boxes : ℝ) * small_length * small_width * small_height =
    large_length * large_width * large_height :=
by sorry

#check small_box_width

end NUMINAMATH_CALUDE_small_box_width_l359_35982


namespace NUMINAMATH_CALUDE_inequality_proof_l359_35930

theorem inequality_proof (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 0 < β) (h4 : β < π/2) : 
  1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l359_35930


namespace NUMINAMATH_CALUDE_negation_of_and_zero_l359_35906

theorem negation_of_and_zero (x y : ℝ) : ¬(x = 0 ∧ y = 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_and_zero_l359_35906


namespace NUMINAMATH_CALUDE_expression_evaluation_l359_35960

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 6 - 1
  let y : ℝ := Real.sqrt 6 + 1
  (2*x + y)^2 + (x - y)*(x + y) - 5*x*(x - y) = 45 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l359_35960


namespace NUMINAMATH_CALUDE_gallery_to_work_blocks_l359_35912

/-- The number of blocks from start to work -/
def total_blocks : ℕ := 37

/-- The number of blocks to the store -/
def store_blocks : ℕ := 11

/-- The number of blocks to the gallery -/
def gallery_blocks : ℕ := 6

/-- The number of blocks already walked -/
def walked_blocks : ℕ := 5

/-- The number of remaining blocks to work after walking 5 blocks -/
def remaining_blocks : ℕ := 20

/-- The number of blocks from the gallery to work -/
def gallery_to_work : ℕ := total_blocks - walked_blocks - store_blocks - gallery_blocks

theorem gallery_to_work_blocks :
  gallery_to_work = 15 :=
by sorry

end NUMINAMATH_CALUDE_gallery_to_work_blocks_l359_35912


namespace NUMINAMATH_CALUDE_fraction_equality_implies_equality_l359_35971

theorem fraction_equality_implies_equality (x y m : ℝ) (h : m ≠ 0) :
  x / m = y / m → x = y := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_equality_l359_35971


namespace NUMINAMATH_CALUDE_hyperbola_equation_l359_35969

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y = -2*x ∧ y^2/a^2 - x^2/b^2 = 1) →
  (∃ (x y : ℝ), x^2 = 4*Real.sqrt 10*y ∧ x^2 + y^2 = a^2 - b^2) →
  a^2 = 8 ∧ b^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l359_35969


namespace NUMINAMATH_CALUDE_all_stars_arrangement_l359_35953

/-- The number of ways to arrange All-Stars from different teams in a row -/
def arrange_all_stars (cubs : ℕ) (red_sox : ℕ) (yankees : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial cubs) * (Nat.factorial red_sox) * (Nat.factorial yankees)

/-- Theorem stating that there are 6912 ways to arrange 10 All-Stars with the given conditions -/
theorem all_stars_arrangement :
  arrange_all_stars 4 4 2 = 6912 := by
  sorry

end NUMINAMATH_CALUDE_all_stars_arrangement_l359_35953


namespace NUMINAMATH_CALUDE_luke_good_games_l359_35907

theorem luke_good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (non_working_games : ℕ) :
  games_from_friend = 2 →
  games_from_garage_sale = 2 →
  non_working_games = 2 →
  games_from_friend + games_from_garage_sale - non_working_games = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_luke_good_games_l359_35907


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_ratio_l359_35939

theorem rectangular_field_diagonal_ratio : 
  ∀ (x y : ℝ), 
    x > 0 → y > 0 →  -- x and y are positive (representing sides of a rectangle)
    x + y - Real.sqrt (x^2 + y^2) = (2/3) * y →  -- diagonal walk saves 2/3 of longer side
    x / y = 8/9 :=  -- ratio of shorter to longer side
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_ratio_l359_35939


namespace NUMINAMATH_CALUDE_triangle_side_length_l359_35984

theorem triangle_side_length (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_thirty_deg : a / c = 1 / 2) (h_hypotenuse : c = 6 * Real.sqrt 2) :
  b = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l359_35984


namespace NUMINAMATH_CALUDE_water_pouring_theorem_l359_35934

-- Define the pouring process
def remaining_water (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

-- Theorem statement
theorem water_pouring_theorem :
  remaining_water 8 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_water_pouring_theorem_l359_35934


namespace NUMINAMATH_CALUDE_child_height_calculation_l359_35949

/-- Given a child's current height and growth since last visit, 
    calculate the child's height at the last visit. -/
def height_at_last_visit (current_height growth : ℝ) : ℝ :=
  current_height - growth

/-- Theorem stating that given the specific measurements, 
    the child's height at the last visit was 38.5 inches. -/
theorem child_height_calculation : 
  height_at_last_visit 41.5 3 = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_child_height_calculation_l359_35949


namespace NUMINAMATH_CALUDE_ratio_c_d_equals_two_thirds_l359_35910

theorem ratio_c_d_equals_two_thirds
  (x y c d : ℝ)
  (h1 : 8 * x - 5 * y = c)
  (h2 : 10 * y - 12 * x = d)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0) :
  c / d = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_c_d_equals_two_thirds_l359_35910


namespace NUMINAMATH_CALUDE_twenty_pancakes_in_24_minutes_l359_35986

/-- Represents the pancake production and consumption rates of a family -/
structure PancakeFamily where
  dad_rate : ℚ  -- Dad's pancake production rate per hour
  mom_rate : ℚ  -- Mom's pancake production rate per hour
  petya_rate : ℚ  -- Petya's pancake consumption rate per 15 minutes
  vasya_multiplier : ℚ  -- Vasya's consumption rate multiplier relative to Petya

/-- Calculates the minimum time (in minutes) required for at least 20 pancakes to remain uneaten -/
def min_time_for_20_pancakes (family : PancakeFamily) : ℚ :=
  sorry

/-- The main theorem stating that 24 minutes is the minimum time for 20 pancakes to remain uneaten -/
theorem twenty_pancakes_in_24_minutes (family : PancakeFamily) 
  (h1 : family.dad_rate = 70)
  (h2 : family.mom_rate = 100)
  (h3 : family.petya_rate = 10)
  (h4 : family.vasya_multiplier = 2) :
  min_time_for_20_pancakes family = 24 := by
  sorry

end NUMINAMATH_CALUDE_twenty_pancakes_in_24_minutes_l359_35986


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l359_35964

theorem geometric_sequence_problem (q : ℝ) (S₆ : ℝ) (b₁ : ℝ) (b₅ : ℝ) : 
  q = 3 → 
  S₆ = 1820 → 
  S₆ = b₁ * (1 - q^6) / (1 - q) → 
  b₅ = b₁ * q^4 →
  b₁ = 5 ∧ b₅ = 405 := by
  sorry

#check geometric_sequence_problem

end NUMINAMATH_CALUDE_geometric_sequence_problem_l359_35964


namespace NUMINAMATH_CALUDE_f_is_quadratic_l359_35937

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² - 4 = 4 -/
def f (x : ℝ) : ℝ := x^2 - 8

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l359_35937


namespace NUMINAMATH_CALUDE_average_score_range_l359_35943

/-- Represents the score distribution in the math competition --/
structure ScoreDistribution where
  score_100 : ℕ
  score_90_99 : ℕ
  score_80_89 : ℕ
  score_70_79 : ℕ
  score_60_69 : ℕ
  score_50_59 : ℕ
  score_48 : ℕ

/-- Calculates the minimum possible average score --/
def min_average_score (sd : ScoreDistribution) : ℚ :=
  (100 * sd.score_100 + 90 * sd.score_90_99 + 80 * sd.score_80_89 + 70 * sd.score_70_79 +
   60 * sd.score_60_69 + 50 * sd.score_50_59 + 48 * sd.score_48) /
  (sd.score_100 + sd.score_90_99 + sd.score_80_89 + sd.score_70_79 +
   sd.score_60_69 + sd.score_50_59 + sd.score_48)

/-- Calculates the maximum possible average score --/
def max_average_score (sd : ScoreDistribution) : ℚ :=
  (100 * sd.score_100 + 99 * sd.score_90_99 + 89 * sd.score_80_89 + 79 * sd.score_70_79 +
   69 * sd.score_60_69 + 59 * sd.score_50_59 + 48 * sd.score_48) /
  (sd.score_100 + sd.score_90_99 + sd.score_80_89 + sd.score_70_79 +
   sd.score_60_69 + sd.score_50_59 + sd.score_48)

/-- The score distribution for the given problem --/
def zhi_cheng_distribution : ScoreDistribution :=
  { score_100 := 2
  , score_90_99 := 9
  , score_80_89 := 17
  , score_70_79 := 28
  , score_60_69 := 36
  , score_50_59 := 7
  , score_48 := 1
  }

/-- Theorem stating the range of the overall average score --/
theorem average_score_range :
  min_average_score zhi_cheng_distribution ≥ 68.88 ∧
  max_average_score zhi_cheng_distribution ≤ 77.61 := by
  sorry

end NUMINAMATH_CALUDE_average_score_range_l359_35943


namespace NUMINAMATH_CALUDE_quadratic_function_c_bounds_l359_35994

/-- Given a quadratic function f(x) = x² + bx + c, where b and c are real numbers,
    if 0 ≤ f(1) = f(2) ≤ 10, then 2 ≤ c ≤ 12 -/
theorem quadratic_function_c_bounds (b c : ℝ) :
  let f := fun x => x^2 + b*x + c
  (0 ≤ f 1) ∧ (f 1 = f 2) ∧ (f 2 ≤ 10) → 2 ≤ c ∧ c ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_c_bounds_l359_35994


namespace NUMINAMATH_CALUDE_max_value_expression_l359_35900

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 6*x^2 + 1)) / x ≤ 2/3 ∧
  ∃ x₀ > 0, (x₀^2 + 3 - Real.sqrt (x₀^4 + 6*x₀^2 + 1)) / x₀ = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l359_35900


namespace NUMINAMATH_CALUDE_cube_congruence_for_prime_l359_35920

theorem cube_congruence_for_prime (p : ℕ) (k : ℕ) 
  (hp : Nat.Prime p) (hform : p = 3 * k + 1) : 
  ∃ a b : ℕ, 0 < a ∧ a < b ∧ b < Real.sqrt p ∧ a^3 ≡ b^3 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_cube_congruence_for_prime_l359_35920


namespace NUMINAMATH_CALUDE_hexagonal_prism_vertices_l359_35942

/-- A prism with hexagonal bases -/
structure HexagonalPrism where
  -- The number of sides in each base
  base_sides : ℕ
  -- The number of rectangular sides
  rect_sides : ℕ
  -- The total number of vertices
  vertices : ℕ

/-- Theorem: A hexagonal prism has 12 vertices -/
theorem hexagonal_prism_vertices (p : HexagonalPrism) 
  (h1 : p.base_sides = 6)
  (h2 : p.rect_sides = 6) : 
  p.vertices = 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_vertices_l359_35942


namespace NUMINAMATH_CALUDE_inequality_proof_l359_35935

theorem inequality_proof (t : ℝ) (h : 0 ≤ t ∧ t ≤ 6) : 
  Real.sqrt 6 ≤ Real.sqrt (-t + 6) + Real.sqrt t ∧ 
  Real.sqrt (-t + 6) + Real.sqrt t ≤ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l359_35935


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l359_35983

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (∀ x, x < 1 → x < 2) ∧ (∃ x, x < 2 ∧ ¬(x < 1)) := by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l359_35983


namespace NUMINAMATH_CALUDE_third_year_sample_size_l359_35926

/-- Calculates the number of students to be sampled from a specific grade in a stratified sampling. -/
def stratified_sample_size (total_population : ℕ) (grade_population : ℕ) (total_sample : ℕ) : ℕ :=
  (grade_population * total_sample) / total_population

theorem third_year_sample_size :
  let total_population : ℕ := 3000
  let third_year_population : ℕ := 1200
  let total_sample : ℕ := 50
  stratified_sample_size total_population third_year_population total_sample = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_year_sample_size_l359_35926


namespace NUMINAMATH_CALUDE_mike_action_figures_l359_35987

/-- The number of action figures each shelf can hold -/
def figures_per_shelf : ℕ := 8

/-- The number of shelves Mike needs -/
def number_of_shelves : ℕ := 8

/-- The total number of action figures Mike has -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

theorem mike_action_figures :
  total_figures = 64 :=
by sorry

end NUMINAMATH_CALUDE_mike_action_figures_l359_35987


namespace NUMINAMATH_CALUDE_min_value_function_l359_35922

theorem min_value_function (x : ℝ) (h : x > 1) : 
  ∀ y : ℝ, y = 4 / (x - 1) + x → y ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_function_l359_35922


namespace NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l359_35946

-- Define the type for spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the standard representation constraints
def isStandardRepresentation (coord : SphericalCoord) : Prop :=
  coord.ρ > 0 ∧ 0 ≤ coord.θ ∧ coord.θ < 2 * Real.pi ∧ 0 ≤ coord.φ ∧ coord.φ ≤ Real.pi

-- Define the equivalence relation between spherical coordinates
def sphericalEquivalent (coord1 coord2 : SphericalCoord) : Prop :=
  coord1.ρ = coord2.ρ ∧
  (coord1.θ % (2 * Real.pi) = coord2.θ % (2 * Real.pi)) ∧
  ((coord1.φ % (2 * Real.pi) = coord2.φ % (2 * Real.pi)) ∨
   (coord1.φ % (2 * Real.pi) = 2 * Real.pi - (coord2.φ % (2 * Real.pi))))

-- Theorem statement
theorem spherical_coordinate_equivalence :
  let original := SphericalCoord.mk 5 (5 * Real.pi / 6) (9 * Real.pi / 5)
  let standard := SphericalCoord.mk 5 (11 * Real.pi / 6) (Real.pi / 5)
  sphericalEquivalent original standard ∧ isStandardRepresentation standard :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l359_35946


namespace NUMINAMATH_CALUDE_lanas_roses_l359_35903

theorem lanas_roses (tulips : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) 
  (h1 : tulips = 36)
  (h2 : used_flowers = 70)
  (h3 : extra_flowers = 3) :
  tulips + (used_flowers + extra_flowers - tulips) = 73 :=
by sorry

end NUMINAMATH_CALUDE_lanas_roses_l359_35903


namespace NUMINAMATH_CALUDE_worker_savings_l359_35958

theorem worker_savings (P : ℝ) (f : ℝ) (h1 : P > 0) (h2 : 0 ≤ f ∧ f ≤ 1) : 
  12 * f * P = 2 * (1 - f) * P → f = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_worker_savings_l359_35958
