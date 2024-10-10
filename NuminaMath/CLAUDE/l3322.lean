import Mathlib

namespace green_balloons_l3322_332278

theorem green_balloons (total : Nat) (red : Nat) (green : Nat) : 
  total = 17 → red = 8 → green = total - red → green = 9 := by
  sorry

end green_balloons_l3322_332278


namespace simons_age_is_45_l3322_332214

/-- Simon's age in 2010, given Jorge's age in 2005 and the age difference between Simon and Jorge -/
def simons_age_2010 (jorges_age_2005 : ℕ) (age_difference : ℕ) : ℕ :=
  jorges_age_2005 + (2010 - 2005) + age_difference

/-- Theorem stating that Simon's age in 2010 is 45 years old -/
theorem simons_age_is_45 :
  simons_age_2010 16 24 = 45 := by
  sorry

end simons_age_is_45_l3322_332214


namespace unique_all_ones_polynomial_l3322_332218

def is_all_ones (n : ℕ) : Prop :=
  ∃ k : ℕ+, n = (10^k.val - 1) / 9

def polynomial_all_ones (P : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, is_all_ones n → is_all_ones (P n)

theorem unique_all_ones_polynomial :
  ∀ P : ℕ → ℕ, polynomial_all_ones P → P = id := by sorry

end unique_all_ones_polynomial_l3322_332218


namespace water_evaporation_rate_l3322_332230

/-- Proves that given a glass filled with 10 ounces of water, and 6% of the water
    evaporating over a 30-day period, the amount of water evaporated each day is 0.02 ounces. -/
theorem water_evaporation_rate (initial_water : ℝ) (days : ℕ) (evaporation_percentage : ℝ) :
  initial_water = 10 →
  days = 30 →
  evaporation_percentage = 6 →
  (initial_water * evaporation_percentage / 100) / days = 0.02 := by
  sorry


end water_evaporation_rate_l3322_332230


namespace min_value_of_f_l3322_332267

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := x^2 + 4*x*y + 5*y^2 - 8*x + 6*y + 2

/-- Theorem stating that the minimum value of f is -7 -/
theorem min_value_of_f :
  ∃ (x y : ℝ), ∀ (a b : ℝ), f x y ≤ f a b ∧ f x y = -7 :=
by sorry

end min_value_of_f_l3322_332267


namespace video_card_upgrade_multiple_l3322_332216

theorem video_card_upgrade_multiple (computer_cost monitor_peripheral_ratio base_video_card_cost total_spent : ℚ) :
  computer_cost = 1500 →
  monitor_peripheral_ratio = 1/5 →
  base_video_card_cost = 300 →
  total_spent = 2100 →
  let monitor_peripheral_cost := computer_cost * monitor_peripheral_ratio
  let total_without_upgrade := computer_cost + monitor_peripheral_cost
  let upgraded_video_card_cost := total_spent - total_without_upgrade
  upgraded_video_card_cost / base_video_card_cost = 1 := by
  sorry

end video_card_upgrade_multiple_l3322_332216


namespace tiles_required_for_room_l3322_332221

theorem tiles_required_for_room (room_length room_width tile_length tile_width : ℚ) :
  room_length = 10 →
  room_width = 15 →
  tile_length = 5 / 12 →
  tile_width = 2 / 3 →
  (room_length * room_width) / (tile_length * tile_width) = 540 :=
by
  sorry

end tiles_required_for_room_l3322_332221


namespace equality_for_all_n_l3322_332239

theorem equality_for_all_n (x y a b : ℝ) 
  (h1 : x + y = a + b) 
  (h2 : x^2 + y^2 = a^2 + b^2) : 
  ∀ n : ℤ, x^n + y^n = a^n + b^n := by sorry

end equality_for_all_n_l3322_332239


namespace probability_not_hearing_favorite_song_l3322_332273

/-- Represents the duration of a song in seconds -/
def SongDuration := ℕ

/-- Represents a playlist of songs -/
def Playlist := List SongDuration

/-- Creates a playlist with 12 songs, where each song is 20 seconds longer than the previous one -/
def createPlaylist : Playlist :=
  List.range 12 |>.map (fun i => 20 * (i + 1))

/-- The duration of the favorite song in seconds -/
def favoriteSongDuration : SongDuration := 4 * 60

/-- The total listening time in seconds -/
def totalListeningTime : SongDuration := 5 * 60

/-- Calculates the probability of not hearing the entire favorite song within the first 5 minutes -/
def probabilityNotHearingFavoriteSong (playlist : Playlist) : ℚ :=
  let totalArrangements := Nat.factorial playlist.length
  let favorableArrangements := 3 * Nat.factorial (playlist.length - 2)
  1 - (favorableArrangements : ℚ) / totalArrangements

theorem probability_not_hearing_favorite_song :
  probabilityNotHearingFavoriteSong createPlaylist = 43 / 44 := by
  sorry

#eval probabilityNotHearingFavoriteSong createPlaylist

end probability_not_hearing_favorite_song_l3322_332273


namespace square_plus_inverse_square_implies_fourth_plus_inverse_fourth_l3322_332228

theorem square_plus_inverse_square_implies_fourth_plus_inverse_fourth (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x^2) = 2 → x^4 + (1/x^4) = 2 := by
  sorry

end square_plus_inverse_square_implies_fourth_plus_inverse_fourth_l3322_332228


namespace tyler_puppies_l3322_332212

/-- Given a person with a certain number of dogs, where each dog has a certain number of puppies,
    calculate the total number of puppies. -/
def total_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) : ℕ :=
  num_dogs * puppies_per_dog

/-- Theorem: A person with 15 dogs, where each dog has 5 puppies, has a total of 75 puppies. -/
theorem tyler_puppies : total_puppies 15 5 = 75 := by
  sorry

end tyler_puppies_l3322_332212


namespace volunteer_assignment_l3322_332257

def number_of_volunteers : ℕ := 6
def number_for_training : ℕ := 4
def number_per_location : ℕ := 2

def select_and_assign (n m k : ℕ) : Prop :=
  ∃ (total : ℕ),
    total = Nat.choose (n - 1) k * Nat.choose (n - k - 1) k +
            Nat.choose (n - 1) 1 * Nat.choose (n - 2) k ∧
    total = 60

theorem volunteer_assignment :
  select_and_assign number_of_volunteers number_for_training number_per_location :=
sorry

end volunteer_assignment_l3322_332257


namespace simone_finish_time_l3322_332241

-- Define the start time
def start_time : Nat := 8 * 60  -- 8:00 AM in minutes since midnight

-- Define the duration of the first two tasks
def first_two_tasks_duration : Nat := 2 * 45

-- Define the break duration
def break_duration : Nat := 15

-- Define the duration of the third task
def third_task_duration : Nat := 2 * 45

-- Define the total duration
def total_duration : Nat := first_two_tasks_duration + break_duration + third_task_duration

-- Define the finish time in minutes since midnight
def finish_time : Nat := start_time + total_duration

-- Theorem to prove
theorem simone_finish_time : 
  finish_time = 11 * 60 + 15  -- 11:15 AM in minutes since midnight
  := by sorry

end simone_finish_time_l3322_332241


namespace isosceles_trapezoid_side_length_l3322_332270

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ
  side : ℝ

/-- The theorem stating the relationship between the trapezoid's properties -/
theorem isosceles_trapezoid_side_length 
  (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 6) 
  (h2 : t.base2 = 12) 
  (h3 : t.area = 36) : 
  t.side = 5 := by
  sorry

#check isosceles_trapezoid_side_length

end isosceles_trapezoid_side_length_l3322_332270


namespace bouncy_balls_cost_l3322_332244

def red_packs : ℕ := 5
def yellow_packs : ℕ := 4
def blue_packs : ℕ := 3

def red_balls_per_pack : ℕ := 18
def yellow_balls_per_pack : ℕ := 15
def blue_balls_per_pack : ℕ := 12

def red_price : ℚ := 3/2
def yellow_price : ℚ := 5/4
def blue_price : ℚ := 1

def red_discount : ℚ := 1/10
def blue_discount : ℚ := 1/20

def total_cost (packs : ℕ) (balls_per_pack : ℕ) (price : ℚ) : ℚ :=
  (packs * balls_per_pack : ℚ) * price

def discounted_cost (cost : ℚ) (discount : ℚ) : ℚ :=
  cost * (1 - discount)

theorem bouncy_balls_cost :
  discounted_cost (total_cost red_packs red_balls_per_pack red_price) red_discount = 243/2 ∧
  total_cost yellow_packs yellow_balls_per_pack yellow_price = 75 ∧
  discounted_cost (total_cost blue_packs blue_balls_per_pack blue_price) blue_discount = 342/10 :=
by sorry

end bouncy_balls_cost_l3322_332244


namespace work_efficiency_l3322_332253

/-- Given a person who takes x days to complete a task, and Tanya who is 25% more efficient
    and takes 12 days to complete the same task, prove that x is equal to 15 days. -/
theorem work_efficiency (x : ℝ) : 
  (∃ (person : ℝ → ℝ) (tanya : ℝ → ℝ), 
    (∀ t, tanya t = 0.75 * person t) ∧ 
    (tanya 12 = person x)) → 
  x = 15 := by
sorry

end work_efficiency_l3322_332253


namespace square_vertex_coordinates_l3322_332271

def is_vertex_of_centered_square (x y : ℤ) : Prop :=
  ∃ (s : ℝ), s > 0 ∧ x^2 + y^2 = 2 * s^2

theorem square_vertex_coordinates :
  ∀ x y : ℤ,
    is_vertex_of_centered_square x y →
    Nat.gcd x.natAbs y.natAbs = 2 →
    2 * (x^2 + y^2) = 10 * Nat.lcm x.natAbs y.natAbs →
    ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by sorry

end square_vertex_coordinates_l3322_332271


namespace number_equation_solution_l3322_332206

theorem number_equation_solution : ∃ n : ℝ, 7 * n = 3 * n + 12 ∧ n = 3 := by
  sorry

end number_equation_solution_l3322_332206


namespace junior_score_l3322_332200

theorem junior_score (n : ℝ) (h_n_pos : n > 0) : 
  let junior_percent : ℝ := 0.2
  let senior_percent : ℝ := 0.8
  let overall_avg : ℝ := 86
  let senior_avg : ℝ := 85
  let junior_count : ℝ := junior_percent * n
  let senior_count : ℝ := senior_percent * n
  let total_score : ℝ := overall_avg * n
  let senior_total_score : ℝ := senior_avg * senior_count
  let junior_total_score : ℝ := total_score - senior_total_score
  junior_total_score / junior_count = 90 :=
by sorry

end junior_score_l3322_332200


namespace discount_profit_theorem_l3322_332295

theorem discount_profit_theorem (cost : ℝ) (h_cost_pos : cost > 0) : 
  let discount_rate : ℝ := 0.1
  let profit_rate_with_discount : ℝ := 0.2
  let selling_price_with_discount : ℝ := (1 - discount_rate) * ((1 + profit_rate_with_discount) * cost)
  let selling_price_without_discount : ℝ := selling_price_with_discount / (1 - discount_rate)
  let profit_without_discount : ℝ := selling_price_without_discount - cost
  let profit_rate_without_discount : ℝ := profit_without_discount / cost
  profit_rate_without_discount = 1/3 := by sorry

end discount_profit_theorem_l3322_332295


namespace p_and_q_implies_p_or_q_l3322_332219

theorem p_and_q_implies_p_or_q (p q : Prop) : (p ∧ q) → (p ∨ q) := by
  sorry

end p_and_q_implies_p_or_q_l3322_332219


namespace hamburger_combinations_l3322_332243

/-- The number of available condiments -/
def num_condiments : ℕ := 9

/-- The number of patty options -/
def num_patty_options : ℕ := 4

/-- The number of possible combinations for condiments -/
def condiment_combinations : ℕ := 2^num_condiments

/-- The total number of different hamburger combinations -/
def total_combinations : ℕ := num_patty_options * condiment_combinations

/-- Theorem stating that the total number of different hamburger combinations is 2048 -/
theorem hamburger_combinations : total_combinations = 2048 := by
  sorry

end hamburger_combinations_l3322_332243


namespace triangle_forming_sets_l3322_332290

/-- A function that checks if three numbers can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of numbers we're checking --/
def sets : List (ℝ × ℝ × ℝ) := [
  (1, 2, 3),
  (2, 3, 4),
  (3, 4, 5),
  (3, 6, 9)
]

/-- The theorem stating which sets can form triangles --/
theorem triangle_forming_sets :
  (∀ (a b c : ℝ), (a, b, c) ∈ sets → can_form_triangle a b c) ↔
  (∃ (a b c : ℝ), (a, b, c) ∈ sets ∧ can_form_triangle a b c ∧ (a, b, c) = (2, 3, 4)) ∧
  (∃ (a b c : ℝ), (a, b, c) ∈ sets ∧ can_form_triangle a b c ∧ (a, b, c) = (3, 4, 5)) ∧
  (∀ (a b c : ℝ), (a, b, c) ∈ sets → (a, b, c) ≠ (2, 3, 4) → (a, b, c) ≠ (3, 4, 5) → ¬can_form_triangle a b c) :=
sorry

end triangle_forming_sets_l3322_332290


namespace range_of_m_for_cubic_equation_l3322_332211

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem range_of_m_for_cubic_equation (m : ℝ) :
  (∃ x ∈ Set.Icc 0 2, f x + m = 0) → m ∈ Set.Icc (-2) 2 := by
  sorry


end range_of_m_for_cubic_equation_l3322_332211


namespace potential_parallel_necessary_not_sufficient_l3322_332220

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate for parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The condition for potential parallelism -/
def potential_parallel_condition (l1 l2 : Line) : Prop :=
  l1.a * l2.b - l2.a * l1.b = 0

/-- Theorem stating that the condition is necessary but not sufficient for parallelism -/
theorem potential_parallel_necessary_not_sufficient :
  (∀ l1 l2 : Line, parallel l1 l2 → potential_parallel_condition l1 l2) ∧
  ¬(∀ l1 l2 : Line, potential_parallel_condition l1 l2 → parallel l1 l2) :=
sorry

end potential_parallel_necessary_not_sufficient_l3322_332220


namespace cone_base_radius_l3322_332229

/-- Given a cone whose lateral surface is a semicircle with radius 2,
    prove that the radius of the base of the cone is 1. -/
theorem cone_base_radius (r : ℝ) (h : r > 0) : r = 1 := by
  sorry

end cone_base_radius_l3322_332229


namespace triangle_lines_l3322_332276

-- Define the triangle ABC
def A : ℝ × ℝ := (3, -4)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (-5, 2)

-- Define the altitude BD
def altitude_BD (x y : ℝ) : Prop := 4 * x - 3 * y - 24 = 0

-- Define the median BE
def median_BE (x y : ℝ) : Prop := x - 7 * y - 6 = 0

-- Theorem statement
theorem triangle_lines :
  (∀ x y : ℝ, altitude_BD x y ↔ 
    (x - B.1) * (C.2 - A.2) = (y - B.2) * (C.1 - A.1)) ∧
  (∀ x y : ℝ, median_BE x y ↔ 
    2 * (x - B.1) = (A.1 + C.1) - 2 * B.1 ∧
    2 * (y - B.2) = (A.2 + C.2) - 2 * B.2) :=
sorry

end triangle_lines_l3322_332276


namespace max_excellent_courses_l3322_332265

/-- A course video with two attributes: number of views and expert score -/
structure CourseVideo where
  views : ℕ
  expertScore : ℕ

/-- Defines when one course video is not inferior to another -/
def notInferior (a b : CourseVideo) : Prop :=
  a.views ≥ b.views ∨ a.expertScore ≥ b.expertScore

/-- Defines an excellent course video -/
def isExcellent (a : CourseVideo) (courses : Finset CourseVideo) : Prop :=
  ∀ b ∈ courses, b ≠ a → notInferior a b

/-- Theorem: It's possible to have 5 excellent course videos among 5 courses -/
theorem max_excellent_courses (courses : Finset CourseVideo) (h : courses.card = 5) :
  ∃ excellentCourses : Finset CourseVideo,
    excellentCourses ⊆ courses ∧
    excellentCourses.card = 5 ∧
    ∀ a ∈ excellentCourses, isExcellent a courses := by
  sorry

end max_excellent_courses_l3322_332265


namespace math_class_size_l3322_332224

theorem math_class_size (total : ℕ) (both : ℕ) :
  total = 75 →
  both = 10 →
  ∃ (math physics : ℕ),
    total = math + physics - both ∧
    math = 2 * physics →
    math = 56 := by
  sorry

end math_class_size_l3322_332224


namespace not_all_perfect_squares_l3322_332259

theorem not_all_perfect_squares (a b c : ℕ+) : 
  ¬(∃ (x y z : ℕ), x^2 = a^2 + b + c ∧ y^2 = b^2 + c + a ∧ z^2 = c^2 + a + b) := by
  sorry

end not_all_perfect_squares_l3322_332259


namespace modular_inverse_89_mod_90_l3322_332254

theorem modular_inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x < 90 ∧ (89 * x) % 90 = 1 := by
  sorry

end modular_inverse_89_mod_90_l3322_332254


namespace jeans_savings_theorem_l3322_332225

/-- Calculates the amount saved on a pair of jeans given the original price and discounts -/
def calculate_savings (original_price : ℝ) (sale_discount_percent : ℝ) (coupon_discount : ℝ) (credit_card_discount_percent : ℝ) : ℝ :=
  let price_after_sale := original_price * (1 - sale_discount_percent)
  let price_after_coupon := price_after_sale - coupon_discount
  let final_price := price_after_coupon * (1 - credit_card_discount_percent)
  original_price - final_price

/-- Theorem stating that the savings on the jeans is $44 -/
theorem jeans_savings_theorem :
  calculate_savings 125 0.20 10 0.10 = 44 := by
  sorry

#eval calculate_savings 125 0.20 10 0.10

end jeans_savings_theorem_l3322_332225


namespace cube_with_holes_surface_area_l3322_332282

/-- Calculates the total surface area of a cube with holes --/
def total_surface_area (cube_edge : ℝ) (hole_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_side^2
  let exposed_area := 6 * 4 * hole_side^2
  original_surface_area - hole_area + exposed_area

/-- Theorem: The total surface area of a cube with edge length 4 meters and square holes
    of side 2 meters cut through each face is 168 square meters --/
theorem cube_with_holes_surface_area :
  total_surface_area 4 2 = 168 := by
  sorry

end cube_with_holes_surface_area_l3322_332282


namespace average_and_difference_l3322_332223

theorem average_and_difference (x : ℝ) : 
  (30 + x) / 2 = 34 → |x - 30| = 8 := by
  sorry

end average_and_difference_l3322_332223


namespace F_value_at_2_l3322_332210

/-- F is a polynomial function of degree 7 -/
def F (a b c d : ℝ) (x : ℝ) : ℝ := a*x^7 + b*x^5 + c*x^3 + d*x - 6

/-- Theorem: Given F(x) = ax^7 + bx^5 + cx^3 + dx - 6 and F(-2) = 10, prove that F(2) = -22 -/
theorem F_value_at_2 (a b c d : ℝ) (h : F a b c d (-2) = 10) : F a b c d 2 = -22 := by
  sorry

end F_value_at_2_l3322_332210


namespace baker_cakes_sold_l3322_332268

theorem baker_cakes_sold (initial_cakes : ℕ) (additional_cakes : ℕ) (remaining_cakes : ℕ) : 
  initial_cakes = 110 →
  additional_cakes = 76 →
  remaining_cakes = 111 →
  initial_cakes + additional_cakes - remaining_cakes = 75 := by
sorry

end baker_cakes_sold_l3322_332268


namespace initial_gathering_size_l3322_332201

theorem initial_gathering_size (initial_snackers : ℕ)
  (h1 : initial_snackers = 100)
  (h2 : ∃ (a b c d e : ℕ),
    a = initial_snackers + 20 ∧
    b = a / 2 + 10 ∧
    c = b - 30 ∧
    d = c / 2 ∧
    d = 20) :
  initial_snackers = 100 := by
sorry

end initial_gathering_size_l3322_332201


namespace hyperbola_real_axis_length_l3322_332236

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 (a > 0), 
    if a right triangle is formed by its left and right foci and the point (2, 1),
    then the length of its real axis is 2. -/
theorem hyperbola_real_axis_length (a : ℝ) (h1 : a > 0) : 
  let f (x y : ℝ) := x^2 / a^2 - y^2 / 4
  ∃ (c : ℝ), (2 - c) * (2 + c) + 1 * 1 = 0 ∧ 2 * a = 2 :=
by sorry

end hyperbola_real_axis_length_l3322_332236


namespace f_properties_l3322_332232

/-- Represents a natural number in base 3 notation -/
structure Base3 where
  digits : List Nat
  first_nonzero : digits.head? ≠ some 0
  all_less_than_3 : ∀ d ∈ digits, d < 3

/-- Converts a natural number to its Base3 representation -/
noncomputable def toBase3 (n : ℕ) : Base3 := sorry

/-- Converts a Base3 representation back to a natural number -/
noncomputable def fromBase3 (b : Base3) : ℕ := sorry

/-- The function f as described in the problem -/
noncomputable def f (n : ℕ) : ℕ :=
  let b := toBase3 n
  match b.digits with
  | 1 :: rest => fromBase3 ⟨2 :: rest, sorry, sorry⟩
  | 2 :: rest => fromBase3 ⟨1 :: (rest ++ [0]), sorry, sorry⟩
  | _ => n  -- This case should not occur for valid Base3 numbers

/-- The main theorem to be proved -/
theorem f_properties :
  (∀ m n, m < n → f m < f n) ∧  -- Strictly monotone
  (∀ n, f (f n) = 3 * n) := by
  sorry


end f_properties_l3322_332232


namespace intersection_of_M_and_N_l3322_332297

open Set

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x < 1}

theorem intersection_of_M_and_N : M ∩ N = Ioo (-1) 1 := by
  sorry

end intersection_of_M_and_N_l3322_332297


namespace unique_b_value_l3322_332202

theorem unique_b_value : ∃! b : ℝ, ∃ x : ℝ, x^2 + b*x + 1 = 0 ∧ x^2 + x + b = 0 ∧ b = -2 := by
  sorry

end unique_b_value_l3322_332202


namespace johns_piggy_bank_l3322_332209

theorem johns_piggy_bank (quarters dimes nickels : ℕ) : 
  quarters + dimes + nickels = 63 →
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters = 22 := by
sorry

end johns_piggy_bank_l3322_332209


namespace max_quotient_four_digit_number_l3322_332245

def is_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

theorem max_quotient_four_digit_number (a b c d : ℕ) 
  (ha : is_digit a) (hb : is_digit b) (hc : is_digit c) (hd : is_digit d)
  (hdiff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (1000 * a + 100 * b + 10 * c + d : ℚ) / (a + b + c + d) ≤ 329.2 := by
  sorry

end max_quotient_four_digit_number_l3322_332245


namespace range_of_a_l3322_332249

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y + 4 = 2*x*y → 
    x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) → 
  a ≤ 17/4 := by
  sorry

end range_of_a_l3322_332249


namespace regression_change_l3322_332277

/-- Represents a linear regression equation of the form ŷ = a + bx -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Calculates the change in y when x increases by one unit -/
def change_in_y (regression : LinearRegression) : ℝ := -regression.b

/-- Theorem: For the given regression equation ŷ = 2 - 1.5x, 
    when x increases by one unit, y decreases by 1.5 units -/
theorem regression_change (regression : LinearRegression) 
  (h1 : regression.a = 2) 
  (h2 : regression.b = -1.5) : 
  change_in_y regression = 1.5 := by
  sorry

end regression_change_l3322_332277


namespace paint_stones_l3322_332215

def canPaintAllBlack (k : Nat) : Prop :=
  1 ≤ k ∧ k ≤ 50 ∧ Nat.gcd 100 (k - 1) = 1

theorem paint_stones (k : Nat) :
  canPaintAllBlack k ↔ ¬∃m : Nat, m ∈ Finset.range 13 ∧ k = 4 * m + 1 :=
by sorry

end paint_stones_l3322_332215


namespace trigonometric_simplification_l3322_332233

theorem trigonometric_simplification (x : ℝ) :
  (2 + 3 * Real.sin x - 4 * Real.cos x) / (2 + 3 * Real.sin x + 4 * Real.cos x) = Real.tan (x / 2) := by
  sorry

end trigonometric_simplification_l3322_332233


namespace smallest_perfect_square_divisible_by_3_and_5_l3322_332286

theorem smallest_perfect_square_divisible_by_3_and_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ 3 ∣ n ∧ 5 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → (∃ j : ℕ, m = j^2) → 3 ∣ m → 5 ∣ m → m ≥ n :=
by
  -- Proof goes here
  sorry

#eval (15 : ℕ)^2  -- Expected output: 225

end smallest_perfect_square_divisible_by_3_and_5_l3322_332286


namespace larry_substitution_l3322_332262

theorem larry_substitution (a b c d f : ℚ) : 
  a = 12 → b = 4 → c = 3 → d = 5 →
  (a / (b / (c * (d - f))) = 12 / 4 / 3 * 5 - f) → f = 5 := by
  sorry

end larry_substitution_l3322_332262


namespace wendys_recycling_points_l3322_332299

/-- Given that Wendy earns 5 points per recycled bag, had 11 bags, and didn't recycle 2 bags,
    prove that she earned 45 points. -/
theorem wendys_recycling_points :
  ∀ (points_per_bag : ℕ) (total_bags : ℕ) (unrecycled_bags : ℕ),
    points_per_bag = 5 →
    total_bags = 11 →
    unrecycled_bags = 2 →
    (total_bags - unrecycled_bags) * points_per_bag = 45 :=
by
  sorry

end wendys_recycling_points_l3322_332299


namespace sqrt_calculations_l3322_332213

theorem sqrt_calculations :
  (2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 3 * Real.sqrt 3) ∧
  ((Real.sqrt 18 - Real.sqrt 3) * Real.sqrt 12 = 6 * Real.sqrt 6 - 6) := by
  sorry

end sqrt_calculations_l3322_332213


namespace shaded_to_white_ratio_is_five_thirds_l3322_332226

/-- A nested square figure where vertices of inner squares are at the midpoints of the sides of the outer squares. -/
structure NestedSquareFigure where
  /-- The number of nested squares in the figure -/
  num_squares : ℕ
  /-- The side length of the outermost square -/
  outer_side_length : ℝ
  /-- Assumption that the figure is constructed with vertices at midpoints -/
  vertices_at_midpoints : Bool

/-- The ratio of the shaded area to the white area in the nested square figure -/
def shaded_to_white_ratio (figure : NestedSquareFigure) : ℚ :=
  5 / 3

/-- Theorem stating that the ratio of shaded to white area is 5/3 -/
theorem shaded_to_white_ratio_is_five_thirds (figure : NestedSquareFigure) 
  (h : figure.vertices_at_midpoints = true) : 
  shaded_to_white_ratio figure = 5 / 3 := by
  sorry

end shaded_to_white_ratio_is_five_thirds_l3322_332226


namespace sarah_interview_count_l3322_332279

theorem sarah_interview_count (oranges pears apples strawberries : ℕ) 
  (h_oranges : oranges = 70)
  (h_pears : pears = 120)
  (h_apples : apples = 147)
  (h_strawberries : strawberries = 113) :
  oranges + pears + apples + strawberries = 450 := by
  sorry

end sarah_interview_count_l3322_332279


namespace sum_product_bounds_l3322_332287

theorem sum_product_bounds (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  -(1/2) ≤ a*b + b*c + c*a ∧ a*b + b*c + c*a ≤ 1 := by
  sorry

end sum_product_bounds_l3322_332287


namespace alexanders_apples_l3322_332203

/-- Prove that Alexander bought 5 apples given the conditions of his shopping trip -/
theorem alexanders_apples : 
  ∀ (apple_price orange_price total_spent num_oranges : ℕ),
    apple_price = 1 →
    orange_price = 2 →
    num_oranges = 2 →
    total_spent = 9 →
    ∃ (num_apples : ℕ), 
      num_apples * apple_price + num_oranges * orange_price = total_spent ∧
      num_apples = 5 := by
  sorry

end alexanders_apples_l3322_332203


namespace dog_bunny_ratio_l3322_332256

/-- Given a total of 375 dogs and bunnies, with 75 dogs, prove that the ratio of dogs to bunnies is 1:4 -/
theorem dog_bunny_ratio (total : ℕ) (dogs : ℕ) (h1 : total = 375) (h2 : dogs = 75) :
  (dogs : ℚ) / (total - dogs : ℚ) = 1 / 4 := by
  sorry

end dog_bunny_ratio_l3322_332256


namespace polar_to_rectangular_l3322_332222

theorem polar_to_rectangular (r θ : ℝ) :
  r = -3 ∧ θ = 5 * π / 6 →
  (r * Real.cos θ = 3 * Real.sqrt 3 / 2) ∧ (r * Real.sin θ = -3 / 2) := by
  sorry

end polar_to_rectangular_l3322_332222


namespace cos_squared_minus_sin_squared_pi_eighth_l3322_332208

theorem cos_squared_minus_sin_squared_pi_eighth :
  Real.cos (π / 8) ^ 2 - Real.sin (π / 8) ^ 2 = Real.sqrt 2 / 2 := by
  sorry

end cos_squared_minus_sin_squared_pi_eighth_l3322_332208


namespace gcd_of_powers_of_47_plus_one_l3322_332217

theorem gcd_of_powers_of_47_plus_one (h : Nat.Prime 47) :
  Nat.gcd (47^6 + 1) (47^6 + 47^3 + 1) = 1 := by
  sorry

end gcd_of_powers_of_47_plus_one_l3322_332217


namespace jane_earnings_l3322_332284

/-- Calculates the earnings from selling eggs over a given number of weeks -/
def egg_earnings (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  let eggs_per_week := num_chickens * eggs_per_chicken
  let dozens_per_week := eggs_per_week / 12
  let earnings_per_week := dozens_per_week * price_per_dozen
  earnings_per_week * num_weeks

/-- Proves that Jane's earnings from selling eggs over two weeks is $20 -/
theorem jane_earnings :
  egg_earnings 10 6 2 2 = 20 := by
  sorry

#eval egg_earnings 10 6 2 2

end jane_earnings_l3322_332284


namespace x_equals_ten_l3322_332255

/-- A structure representing the number pyramid --/
structure NumberPyramid where
  row1_left : ℕ
  row1_right : ℕ
  row2_left : ℕ
  row2_middle : ℕ → ℕ
  row2_right : ℕ → ℕ
  row3_left : ℕ → ℕ
  row3_right : ℕ → ℕ
  row4 : ℕ → ℕ

/-- The theorem stating that x must be 10 given the conditions --/
theorem x_equals_ten (pyramid : NumberPyramid) 
  (h1 : pyramid.row1_left = 11)
  (h2 : pyramid.row1_right = 49)
  (h3 : pyramid.row2_left = 11)
  (h4 : ∀ x, pyramid.row2_middle x = 6 + x)
  (h5 : ∀ x, pyramid.row2_right x = x + 7)
  (h6 : ∀ x, pyramid.row3_left x = pyramid.row2_left + pyramid.row2_middle x)
  (h7 : ∀ x, pyramid.row3_right x = pyramid.row2_middle x + pyramid.row2_right x)
  (h8 : ∀ x, pyramid.row4 x = pyramid.row3_left x + pyramid.row3_right x)
  (h9 : pyramid.row4 10 = 60) :
  ∃ x, x = 10 ∧ pyramid.row4 x = 60 :=
sorry

end x_equals_ten_l3322_332255


namespace even_function_value_l3322_332269

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_value (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : ∀ x, f (x + 2) * f x = 4)
  (h_positive : ∀ x, f x > 0) :
  f 2017 = 2 := by sorry

end even_function_value_l3322_332269


namespace mary_needs_30_apples_l3322_332296

/-- Calculates the number of additional apples needed for baking pies -/
def additional_apples_needed (num_pies : ℕ) (apples_per_pie : ℕ) (apples_harvested : ℕ) : ℕ :=
  max ((num_pies * apples_per_pie) - apples_harvested) 0

/-- Proves that Mary needs to buy 30 more apples -/
theorem mary_needs_30_apples : additional_apples_needed 10 8 50 = 30 := by
  sorry

end mary_needs_30_apples_l3322_332296


namespace gunther_free_time_l3322_332205

/-- Represents the time in minutes for each cleaning task and the total free time --/
structure CleaningTime where
  vacuuming : ℕ
  dusting : ℕ
  mopping : ℕ
  brushing_per_cat : ℕ
  num_cats : ℕ
  free_time : ℕ

/-- Calculates the remaining free time after cleaning --/
def remaining_free_time (ct : CleaningTime) : ℕ :=
  ct.free_time - (ct.vacuuming + ct.dusting + ct.mopping + ct.brushing_per_cat * ct.num_cats)

/-- Theorem stating that Gunther will have 30 minutes of free time left --/
theorem gunther_free_time :
  let ct : CleaningTime := {
    vacuuming := 45,
    dusting := 60,
    mopping := 30,
    brushing_per_cat := 5,
    num_cats := 3,
    free_time := 180
  }
  remaining_free_time ct = 30 := by
  sorry


end gunther_free_time_l3322_332205


namespace find_set_C_l3322_332204

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

theorem find_set_C : 
  ∃ C : Set ℝ, 
    (C = {0, 1, 2}) ∧ 
    (∀ a : ℝ, a ∈ C ↔ A ∪ B a = A) :=
by sorry

end find_set_C_l3322_332204


namespace quadratic_roots_coefficients_l3322_332275

theorem quadratic_roots_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = 2) →
  b = -3 ∧ c = 2 := by
sorry

end quadratic_roots_coefficients_l3322_332275


namespace cube_root_sum_equals_two_l3322_332260

theorem cube_root_sum_equals_two :
  ∃ n : ℤ, (Real.rpow (2 + 10/9 * Real.sqrt 3) (1/3 : ℝ) + Real.rpow (2 - 10/9 * Real.sqrt 3) (1/3 : ℝ) = n) →
  Real.rpow (2 + 10/9 * Real.sqrt 3) (1/3 : ℝ) + Real.rpow (2 - 10/9 * Real.sqrt 3) (1/3 : ℝ) = 2 :=
by sorry

end cube_root_sum_equals_two_l3322_332260


namespace certain_number_problem_l3322_332285

theorem certain_number_problem (x : ℤ) (h : x + 14 = 56) : 3 * x = 126 := by
  sorry

end certain_number_problem_l3322_332285


namespace total_amount_correct_l3322_332238

/-- The rate for painting fences in dollars per meter -/
def painting_rate : ℚ := 0.20

/-- The number of fences to be painted -/
def number_of_fences : ℕ := 50

/-- The length of each fence in meters -/
def fence_length : ℕ := 500

/-- The total amount earned from painting all fences -/
def total_amount : ℚ := 5000

/-- Theorem stating that the total amount earned is correct given the conditions -/
theorem total_amount_correct : 
  painting_rate * (number_of_fences * fence_length : ℚ) = total_amount := by
  sorry

end total_amount_correct_l3322_332238


namespace tetrahedron_sum_l3322_332283

-- Define the tetrahedron with four positive integers on its faces
def Tetrahedron (a b c d : ℕ+) : Prop :=
  -- The sum of the products of each combination of three numbers is 770
  a.val * b.val * c.val + a.val * b.val * d.val + a.val * c.val * d.val + b.val * c.val * d.val = 770

-- Theorem statement
theorem tetrahedron_sum (a b c d : ℕ+) (h : Tetrahedron a b c d) :
  a.val + b.val + c.val + d.val = 57 := by
  sorry

end tetrahedron_sum_l3322_332283


namespace residue_of_negative_1235_mod_29_l3322_332280

theorem residue_of_negative_1235_mod_29 : Int.mod (-1235) 29 = 12 := by
  sorry

end residue_of_negative_1235_mod_29_l3322_332280


namespace quadratic_inequality_l3322_332235

theorem quadratic_inequality (x : ℝ) : -3 * x^2 - 9 * x - 6 ≥ -12 ↔ -2 ≤ x ∧ x ≤ 1 := by
  sorry

end quadratic_inequality_l3322_332235


namespace min_production_avoids_loss_less_than_min_production_incurs_loss_l3322_332274

/-- The minimum production quantity to avoid a loss -/
def min_production : ℝ := 150

/-- The unit selling price in million yuan -/
def unit_price : ℝ := 0.25

/-- The total cost function in million yuan for x units -/
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The total revenue function in million yuan for x units -/
def total_revenue (x : ℝ) : ℝ := unit_price * x

/-- Theorem stating that the minimum production quantity to avoid a loss is 150 units -/
theorem min_production_avoids_loss :
  ∀ x : ℝ, x ≥ min_production → total_revenue x ≥ total_cost x :=
by
  sorry

/-- Theorem stating that any production quantity less than 150 units results in a loss -/
theorem less_than_min_production_incurs_loss :
  ∀ x : ℝ, 0 ≤ x ∧ x < min_production → total_revenue x < total_cost x :=
by
  sorry

end min_production_avoids_loss_less_than_min_production_incurs_loss_l3322_332274


namespace aquarium_visitors_not_ill_l3322_332240

theorem aquarium_visitors_not_ill (total_visitors : ℕ) (ill_percentage : ℚ) : 
  total_visitors = 500 → 
  ill_percentage = 40 / 100 → 
  total_visitors - (total_visitors * ill_percentage).floor = 300 := by
sorry

end aquarium_visitors_not_ill_l3322_332240


namespace naoh_combined_is_54_l3322_332291

/-- Represents the balanced chemical equation coefficients -/
structure BalancedEquation :=
  (naoh_coeff : ℕ)
  (h2so4_coeff : ℕ)
  (h2o_coeff : ℕ)

/-- Represents the given information about the reaction -/
structure ReactionInfo :=
  (h2so4_available : ℕ)
  (h2o_formed : ℕ)
  (equation : BalancedEquation)

/-- Calculates the number of moles of NaOH combined in the reaction -/
def naoh_combined (info : ReactionInfo) : ℕ :=
  info.h2o_formed * info.equation.naoh_coeff / info.equation.h2o_coeff

/-- Theorem stating that given the reaction information, 54 moles of NaOH were combined -/
theorem naoh_combined_is_54 (info : ReactionInfo) 
  (h_h2so4 : info.h2so4_available = 3)
  (h_h2o : info.h2o_formed = 54)
  (h_eq : info.equation = {naoh_coeff := 2, h2so4_coeff := 1, h2o_coeff := 2}) :
  naoh_combined info = 54 := by
  sorry

end naoh_combined_is_54_l3322_332291


namespace expression_evaluation_l3322_332288

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  (2*x + y) * (2*x - y) - 3*(2*x^2 - x*y) + y^2 = -14 :=
by sorry

end expression_evaluation_l3322_332288


namespace potato_price_is_one_l3322_332247

def initial_money : ℚ := 60
def celery_price : ℚ := 5
def cereal_price : ℚ := 12
def cereal_discount : ℚ := 0.5
def bread_price : ℚ := 8
def milk_price : ℚ := 10
def milk_discount : ℚ := 0.1
def num_potatoes : ℕ := 6
def money_left : ℚ := 26

def discounted_price (price : ℚ) (discount : ℚ) : ℚ :=
  price * (1 - discount)

theorem potato_price_is_one :
  let celery_cost := celery_price
  let cereal_cost := discounted_price cereal_price cereal_discount
  let bread_cost := bread_price
  let milk_cost := discounted_price milk_price milk_discount
  let total_cost := celery_cost + cereal_cost + bread_cost + milk_cost
  let potato_coffee_cost := initial_money - money_left
  let potato_cost := potato_coffee_cost - total_cost
  potato_cost / num_potatoes = 1 := by sorry

end potato_price_is_one_l3322_332247


namespace min_visible_pairs_155_birds_l3322_332298

/-- The number of birds on the circle -/
def num_birds : ℕ := 155

/-- The visibility threshold in degrees -/
def visibility_threshold : ℝ := 10

/-- A function that calculates the minimum number of mutually visible bird pairs -/
def min_visible_pairs (n : ℕ) (threshold : ℝ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of mutually visible bird pairs -/
theorem min_visible_pairs_155_birds :
  min_visible_pairs num_birds visibility_threshold = 270 :=
sorry

end min_visible_pairs_155_birds_l3322_332298


namespace parallel_line_existence_l3322_332264

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define parallelism
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∧ l1.b ≠ 0 ∧ l2.a ≠ 0 ∧ l2.b ≠ 0

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem parallel_line_existence (A : Point) (l : Line) :
  ∃ (m : Line), passes_through m A ∧ parallel m l :=
sorry

end parallel_line_existence_l3322_332264


namespace no_141_cents_combination_l3322_332231

/-- Represents the different types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.HalfDollar => 50

/-- Represents a selection of three coins --/
structure CoinSelection :=
  (coin1 : Coin)
  (coin2 : Coin)
  (coin3 : Coin)

/-- Calculates the total value of a coin selection in cents --/
def totalValue (selection : CoinSelection) : Nat :=
  coinValue selection.coin1 + coinValue selection.coin2 + coinValue selection.coin3

/-- Theorem stating that no combination of three coins can sum to 141 cents --/
theorem no_141_cents_combination :
  ∀ (selection : CoinSelection), totalValue selection ≠ 141 := by
  sorry

end no_141_cents_combination_l3322_332231


namespace k_range_l3322_332251

-- Define the hyperbola equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 4) + y^2 / (k - 6) = 1

-- Define the ellipse equation
def ellipse (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / 5 + y^2 / k = 1

-- Define the condition for the line through M(2,1) intersecting the ellipse
def line_intersects_ellipse (k : ℝ) : Prop :=
  ∀ m b : ℝ, ∃ x y : ℝ, y = m * x + b ∧ ellipse k x y ∧ (2 * m + b = 1)

-- Main theorem
theorem k_range :
  (∀ k : ℝ, is_hyperbola k → line_intersects_ellipse k → k > 5 ∧ k < 6) ∧
  (∀ k : ℝ, k > 5 ∧ k < 6 → is_hyperbola k ∧ line_intersects_ellipse k) :=
sorry

end k_range_l3322_332251


namespace quilt_shaded_fraction_l3322_332227

/-- Represents a square quilt block -/
structure QuiltBlock where
  totalSquares : ℕ
  dividedSquares : ℕ
  shadePerDividedSquare : ℚ

/-- The fraction of the quilt block that is shaded -/
def shadedFraction (q : QuiltBlock) : ℚ :=
  (q.dividedSquares : ℚ) * q.shadePerDividedSquare / q.totalSquares

/-- Theorem stating that for a quilt block with 16 total squares, 
    4 divided squares, and half of each divided square shaded,
    the shaded fraction is 1/8 -/
theorem quilt_shaded_fraction :
  ∀ (q : QuiltBlock), 
    q.totalSquares = 16 → 
    q.dividedSquares = 4 → 
    q.shadePerDividedSquare = 1/2 →
    shadedFraction q = 1/8 := by
  sorry

end quilt_shaded_fraction_l3322_332227


namespace triangle_angle_B_l3322_332281

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides

-- Define the theorem
theorem triangle_angle_B (t : Triangle) :
  t.A = π/4 ∧ t.a = Real.sqrt 2 ∧ t.b = Real.sqrt 3 →
  t.B = π/3 ∨ t.B = 2*π/3 :=
by
  sorry

end triangle_angle_B_l3322_332281


namespace equation_solution_l3322_332250

theorem equation_solution (p m z : ℤ) : 
  Prime p ∧ m > 0 ∧ z < 0 ∧ p^3 + p*m + 2*z*m = m^2 + p*z + z^2 ↔ 
  (p = 2 ∧ m = 4 + z ∧ (z = -1 ∨ z = -2 ∨ z = -3)) := by
  sorry

end equation_solution_l3322_332250


namespace yuan_jiao_conversion_meter_cm_conversion_l3322_332272

-- Define the conversion rates
def jiao_per_yuan : ℚ := 10
def cm_per_meter : ℚ := 100

-- Define the conversion functions
def jiao_to_yuan (j : ℚ) : ℚ := j / jiao_per_yuan
def meters_to_cm (m : ℚ) : ℚ := m * cm_per_meter

-- State the theorems
theorem yuan_jiao_conversion :
  5 + jiao_to_yuan 5 = 5.05 := by sorry

theorem meter_cm_conversion :
  meters_to_cm (12 * 0.1) = 120 := by sorry

end yuan_jiao_conversion_meter_cm_conversion_l3322_332272


namespace complex_number_purely_imaginary_l3322_332263

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_purely_imaginary (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - m) m
  is_purely_imaginary z → m = 1 := by
sorry

end complex_number_purely_imaginary_l3322_332263


namespace complementary_of_same_angle_are_equal_l3322_332237

/-- Two angles are complementary if their sum is equal to a right angle (90°) -/
def Complementary (α β : Real) : Prop := α + β = Real.pi / 2

/-- An angle is complementary to itself if it is half of a right angle -/
def SelfComplementary (α : Real) : Prop := α = Real.pi / 4

theorem complementary_of_same_angle_are_equal (α : Real) (h : SelfComplementary α) :
  ∃ β, Complementary α β ∧ α = β := by
  sorry

end complementary_of_same_angle_are_equal_l3322_332237


namespace money_exchange_equations_l3322_332207

/-- Represents the money exchange problem from "Nine Chapters on the Mathematical Art" --/
theorem money_exchange_equations (x y : ℝ) : 
  (x + 1/2 * y = 50 ∧ y + 2/3 * x = 50) ↔ 
  (∃ (a b : ℝ), 
    a = x ∧ 
    b = y ∧ 
    a + 1/2 * b = 50 ∧ 
    b + 2/3 * a = 50) :=
by sorry

end money_exchange_equations_l3322_332207


namespace fifth_term_of_sequence_l3322_332294

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 21 = 40 →
  arithmetic_sequence a₁ d 22 = 44 →
  arithmetic_sequence a₁ d 5 = -24 := by
sorry

end fifth_term_of_sequence_l3322_332294


namespace function_value_at_one_l3322_332234

theorem function_value_at_one (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x - x^2| ≤ 1/4)
  (h2 : ∀ x, |f x + 1 - x^2| ≤ 3/4) : 
  f 1 = 3/4 := by
  sorry

end function_value_at_one_l3322_332234


namespace sum_of_tens_for_hundred_to_ten_l3322_332248

theorem sum_of_tens_for_hundred_to_ten (n : ℕ) : (100 ^ 10) = n * 10 → n = 10 ^ 19 := by
  sorry

end sum_of_tens_for_hundred_to_ten_l3322_332248


namespace last_element_proof_l3322_332289

def first_row (n : ℕ) : ℕ := 2*n - 1

def third_row (n : ℕ) : ℕ := (first_row n) * (first_row n)^2 - (first_row n)

theorem last_element_proof : third_row 5 = 720 := by
  sorry

end last_element_proof_l3322_332289


namespace sequence_max_value_l3322_332252

def x (n : ℕ) : ℚ := (n - 1 : ℚ) / ((n : ℚ)^2 + 1)

theorem sequence_max_value :
  (∀ n : ℕ, x n ≤ (1 : ℚ) / 5) ∧
  x 2 = (1 : ℚ) / 5 ∧
  x 3 = (1 : ℚ) / 5 :=
sorry

end sequence_max_value_l3322_332252


namespace cos_210_degrees_l3322_332246

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l3322_332246


namespace line_contains_point_l3322_332292

/-- Given a line represented by the equation -3/4 - 3kx = 7y that contains the point (1/3, -8),
    prove that k = 55.25 -/
theorem line_contains_point (k : ℝ) : 
  (-3/4 : ℝ) - 3 * k * (1/3 : ℝ) = 7 * (-8 : ℝ) → k = 55.25 := by
  sorry

end line_contains_point_l3322_332292


namespace expression_simplification_l3322_332261

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 + 2 * x) / ((x - 1) * (x + 3)) - (5 * x + 3) / ((x - 1) * (x + 3)) =
  3 * (x^2 - x - 1) / ((x - 1) * (x + 3)) := by
  sorry

end expression_simplification_l3322_332261


namespace sqrt5_diamond_sqrt5_equals_20_l3322_332242

-- Define the custom operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt5_diamond_sqrt5_equals_20 : diamond (Real.sqrt 5) (Real.sqrt 5) = 20 := by
  sorry

end sqrt5_diamond_sqrt5_equals_20_l3322_332242


namespace log_simplification_l3322_332293

-- Define variables
variable (p q r s t z : ℝ)
variable (h₁ : p > 0)
variable (h₂ : q > 0)
variable (h₃ : r > 0)
variable (h₄ : s > 0)
variable (h₅ : t > 0)
variable (h₆ : z > 0)

-- State the theorem
theorem log_simplification :
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * t / (s * z)) = Real.log (z / t) :=
by sorry

end log_simplification_l3322_332293


namespace sum_three_numbers_l3322_332258

theorem sum_three_numbers (a b c N : ℝ) 
  (sum_eq : a + b + c = 105)
  (a_eq : a - 5 = N)
  (b_eq : b + 10 = N)
  (c_eq : 5 * c = N) : 
  N = 50 := by
sorry

end sum_three_numbers_l3322_332258


namespace katie_candy_count_l3322_332266

theorem katie_candy_count (sister_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) 
  (h1 : sister_candy = 23)
  (h2 : eaten_candy = 8)
  (h3 : remaining_candy = 23) :
  ∃ (katie_candy : ℕ), katie_candy = 8 ∧ katie_candy + sister_candy - eaten_candy = remaining_candy :=
by sorry

end katie_candy_count_l3322_332266
