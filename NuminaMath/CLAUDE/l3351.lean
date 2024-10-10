import Mathlib

namespace amoeba_growth_l3351_335115

/-- The population of amoebas after a given number of 10-minute increments -/
def amoeba_population (initial_population : ℕ) (increments : ℕ) : ℕ :=
  initial_population * (3 ^ increments)

/-- Theorem: The amoeba population after 1 hour (6 increments) is 36450 -/
theorem amoeba_growth : amoeba_population 50 6 = 36450 := by
  sorry

end amoeba_growth_l3351_335115


namespace hexagon_angle_sequences_l3351_335135

/-- Represents a sequence of 6 integers for hexagon interior angles -/
def HexagonAngles := (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)

/-- Checks if a sequence of angles is valid according to the problem conditions -/
def is_valid_sequence (angles : HexagonAngles) : Prop :=
  let (a₁, a₂, a₃, a₄, a₅, a₆) := angles
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 720) ∧ 
  (30 ≤ a₁) ∧
  (∀ i, i ∈ [a₁, a₂, a₃, a₄, a₅, a₆] → i < 160) ∧
  (a₁ < a₂) ∧ (a₂ < a₃) ∧ (a₃ < a₄) ∧ (a₄ < a₅) ∧ (a₅ < a₆) ∧
  (∃ d : ℕ, d > 0 ∧ a₂ = a₁ + d ∧ a₃ = a₂ + d ∧ a₄ = a₃ + d ∧ a₅ = a₄ + d ∧ a₆ = a₅ + d)

/-- The main theorem stating that there are exactly 4 valid sequences -/
theorem hexagon_angle_sequences :
  ∃! (sequences : Finset HexagonAngles),
    sequences.card = 4 ∧
    (∀ seq ∈ sequences, is_valid_sequence seq) ∧
    (∀ seq, is_valid_sequence seq → seq ∈ sequences) :=
sorry

end hexagon_angle_sequences_l3351_335135


namespace polynomial_roots_l3351_335127

theorem polynomial_roots : 
  let p (x : ℝ) := x^4 - 2*x^3 - 7*x^2 + 14*x - 6
  ∃ (a b c d : ℝ), 
    (a = 1 ∧ b = 2 ∧ c = (-1 + Real.sqrt 13) / 2 ∧ d = (-1 - Real.sqrt 13) / 2) ∧
    (∀ x : ℝ, p x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end polynomial_roots_l3351_335127


namespace number_difference_l3351_335105

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 41402)
  (b_div_100 : 100 ∣ b)
  (a_eq_b_div_100 : a = b / 100) :
  b - a = 40590 :=
sorry

end number_difference_l3351_335105


namespace food_supply_duration_l3351_335199

/-- Proves that the initial food supply was planned to last for 22 days given the problem conditions -/
theorem food_supply_duration (initial_men : ℕ) (additional_men : ℕ) (remaining_days : ℕ) : 
  initial_men = 760 → 
  additional_men = 2280 → 
  remaining_days = 5 → 
  (initial_men * (22 - 2) : ℕ) = (initial_men + additional_men) * remaining_days :=
by sorry

end food_supply_duration_l3351_335199


namespace radius_is_3_sqrt_13_l3351_335142

/-- Represents a circular sector with an inscribed rectangle -/
structure CircularSectorWithRectangle where
  /-- Radius of the circle -/
  radius : ℝ
  /-- Central angle of the sector in radians -/
  centralAngle : ℝ
  /-- Length of the shorter side of the rectangle -/
  shortSide : ℝ
  /-- Length of the longer side of the rectangle -/
  longSide : ℝ
  /-- The longer side is 3 units longer than the shorter side -/
  sideDifference : longSide = shortSide + 3
  /-- The area of the rectangle is 18 -/
  rectangleArea : shortSide * longSide = 18
  /-- The central angle is 45 degrees (π/4 radians) -/
  angleIs45Degrees : centralAngle = Real.pi / 4

/-- The main theorem stating that the radius is 3√13 -/
theorem radius_is_3_sqrt_13 (sector : CircularSectorWithRectangle) :
  sector.radius = 3 * Real.sqrt 13 := by
  sorry

end radius_is_3_sqrt_13_l3351_335142


namespace landscape_length_is_120_l3351_335144

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  playgroundArea : ℝ
  playgroundRatio : ℝ

/-- The length of the landscape is 4 times its breadth -/
def Landscape.length (l : Landscape) : ℝ := 4 * l.breadth

/-- The total area of the landscape -/
def Landscape.totalArea (l : Landscape) : ℝ := l.length * l.breadth

/-- Theorem: Given a landscape with specific properties, its length is 120 meters -/
theorem landscape_length_is_120 (l : Landscape) 
    (h1 : l.playgroundArea = 1200)
    (h2 : l.playgroundRatio = 1/3)
    (h3 : l.playgroundArea = l.playgroundRatio * l.totalArea) : 
  l.length = 120 := by
  sorry

#check landscape_length_is_120

end landscape_length_is_120_l3351_335144


namespace five_rows_with_seven_students_l3351_335166

/-- Represents the seating arrangement in a classroom -/
structure Seating :=
  (rows_with_7 : ℕ)
  (rows_with_6 : ℕ)

/-- Checks if a seating arrangement is valid -/
def is_valid_seating (s : Seating) : Prop :=
  s.rows_with_7 * 7 + s.rows_with_6 * 6 = 53

/-- The theorem stating that there are 5 rows with 7 students -/
theorem five_rows_with_seven_students :
  ∃ (s : Seating), is_valid_seating s ∧ s.rows_with_7 = 5 :=
sorry

end five_rows_with_seven_students_l3351_335166


namespace fraction_inequality_solution_l3351_335103

theorem fraction_inequality_solution (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 4) ≥ 1 ↔ 
  x < -4 ∨ (-2 < x ∧ x < -Real.sqrt 8) ∨ x > Real.sqrt 8 :=
by sorry

end fraction_inequality_solution_l3351_335103


namespace price_difference_l3351_335184

/-- The price difference problem -/
theorem price_difference (discount_price : ℝ) (discount_rate : ℝ) (increase_rate : ℝ) : 
  discount_price = 68 ∧ 
  discount_rate = 0.15 ∧ 
  increase_rate = 0.25 →
  ∃ (original_price final_price : ℝ),
    original_price * (1 - discount_rate) = discount_price ∧
    final_price = discount_price * (1 + increase_rate) ∧
    final_price - original_price = 5 := by
  sorry

end price_difference_l3351_335184


namespace average_age_increase_l3351_335147

theorem average_age_increase (initial_count : ℕ) (replaced_age1 replaced_age2 : ℕ) 
  (new_average : ℕ) (h1 : initial_count = 8) (h2 : replaced_age1 = 21) 
  (h3 : replaced_age2 = 23) (h4 : new_average = 30) : 
  let initial_total := initial_count * (initial_count * A - replaced_age1 - replaced_age2) / initial_count
  let new_total := initial_total - replaced_age1 - replaced_age2 + 2 * new_average
  let new_average := new_total / initial_count
  new_average - (initial_total / initial_count) = 2 := by
  sorry

end average_age_increase_l3351_335147


namespace division_equation_proof_l3351_335143

theorem division_equation_proof : (320 : ℝ) / (54 + 26) = 4 := by
  sorry

end division_equation_proof_l3351_335143


namespace chessboard_diagonal_ratio_l3351_335119

/-- Represents a rectangle with chessboard coloring -/
structure ChessboardRectangle where
  a : ℕ  -- length
  b : ℕ  -- width

/-- Calculates the ratio of white to black segment lengths on the diagonal -/
def diagonalSegmentRatio (rect : ChessboardRectangle) : ℚ :=
  if rect.a = 100 ∧ rect.b = 99 then 1
  else if rect.a = 101 ∧ rect.b = 99 then 5000 / 4999
  else 0  -- undefined for other dimensions

theorem chessboard_diagonal_ratio :
  ∀ (rect : ChessboardRectangle),
    (rect.a = 100 ∧ rect.b = 99 → diagonalSegmentRatio rect = 1) ∧
    (rect.a = 101 ∧ rect.b = 99 → diagonalSegmentRatio rect = 5000 / 4999) :=
by sorry

end chessboard_diagonal_ratio_l3351_335119


namespace camera_pics_count_l3351_335193

/-- Represents the number of pictures in Olivia's photo collection. -/
structure PhotoCollection where
  phone_pics : ℕ
  camera_pics : ℕ
  albums : ℕ
  pics_per_album : ℕ

/-- The properties of Olivia's photo collection as described in the problem. -/
def olivias_collection : PhotoCollection where
  phone_pics := 5
  camera_pics := 35  -- This is what we want to prove
  albums := 8
  pics_per_album := 5

/-- Theorem stating that the number of pictures from Olivia's camera is 35. -/
theorem camera_pics_count (p : PhotoCollection) 
  (h1 : p.phone_pics = 5)
  (h2 : p.albums = 8)
  (h3 : p.pics_per_album = 5)
  (h4 : p.phone_pics + p.camera_pics = p.albums * p.pics_per_album) :
  p.camera_pics = 35 := by
  sorry

#check camera_pics_count olivias_collection

end camera_pics_count_l3351_335193


namespace function_always_positive_implies_x_range_l3351_335180

theorem function_always_positive_implies_x_range (f : ℝ → ℝ) :
  (∀ a ∈ Set.Icc (-1) 1, ∀ x, f x = x^2 + (a - 4)*x + 4 - 2*a) →
  (∀ a ∈ Set.Icc (-1) 1, ∀ x, f x > 0) →
  ∀ x, f x > 0 → x ∈ Set.union (Set.Iio 1) (Set.Ioi 3) :=
by sorry

end function_always_positive_implies_x_range_l3351_335180


namespace consecutive_numbers_square_l3351_335107

theorem consecutive_numbers_square (a : ℕ) : 
  let b := a + 1
  let c := a * b
  let x := a^2 + b^2 + c^2
  ∃ (k : ℕ), x = (2*k + 1)^2 :=
by sorry

end consecutive_numbers_square_l3351_335107


namespace daves_apps_l3351_335194

theorem daves_apps (initial_files : ℕ) (final_apps : ℕ) (final_files : ℕ) (deleted_apps : ℕ) :
  initial_files = 77 →
  final_apps = 5 →
  final_files = 23 →
  deleted_apps = 11 →
  final_apps + deleted_apps = 16 := by
  sorry

end daves_apps_l3351_335194


namespace algebraic_simplification_and_evaluation_l3351_335167

theorem algebraic_simplification_and_evaluation (a b : ℝ) :
  2 * (a * b^2 + 3 * a^2 * b) - 3 * (a * b^2 + a^2 * b) = -a * b^2 + 3 * a^2 * b ∧
  2 * ((-1) * 2^2 + 3 * (-1)^2 * 2) - 3 * ((-1) * 2^2 + (-1)^2 * 2) = 10 :=
by sorry

end algebraic_simplification_and_evaluation_l3351_335167


namespace expression_simplification_l3351_335109

theorem expression_simplification (x y : ℝ) :
  8 * x + 3 * y - 2 * x + y + 20 + 15 = 6 * x + 4 * y + 35 := by
  sorry

end expression_simplification_l3351_335109


namespace unsold_books_l3351_335186

theorem unsold_books (total : ℕ) (sold : ℕ) (price : ℕ) (revenue : ℕ) : 
  (2 : ℕ) * sold = 3 * total ∧ 
  price = 4 ∧ 
  revenue = 288 ∧ 
  sold * price = revenue → 
  total - sold = 36 := by
  sorry

end unsold_books_l3351_335186


namespace juice_mixture_solution_l3351_335187

/-- Represents the juice mixture problem -/
def JuiceMixture (super_cost mixed_cost acai_cost : ℝ) (acai_amount : ℝ) : Prop :=
  ∃ (mixed_amount : ℝ),
    mixed_amount ≥ 0 ∧
    super_cost * (mixed_amount + acai_amount) =
      mixed_cost * mixed_amount + acai_cost * acai_amount

/-- The solution to the juice mixture problem is approximately 35 litres -/
theorem juice_mixture_solution :
  JuiceMixture 1399.45 262.85 3104.35 23.333333333333336 →
  ∃ (mixed_amount : ℝ),
    mixed_amount ≥ 0 ∧
    abs (mixed_amount - 35) < 0.01 :=
by sorry

end juice_mixture_solution_l3351_335187


namespace songs_added_l3351_335173

theorem songs_added (initial : ℕ) (deleted : ℕ) (final : ℕ) 
  (h1 : initial = 30) 
  (h2 : deleted = 8) 
  (h3 : final = 32) : 
  final - (initial - deleted) = 10 := by
  sorry

end songs_added_l3351_335173


namespace sophie_laundry_loads_l3351_335149

/-- Represents the cost of a box of dryer sheets in dollars -/
def box_cost : ℚ := 5.5

/-- Represents the number of dryer sheets in a box -/
def sheets_per_box : ℕ := 104

/-- Represents the amount saved in a year by not buying dryer sheets, in dollars -/
def yearly_savings : ℚ := 11

/-- Represents the number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Represents the number of dryer sheets used per load of laundry -/
def sheets_per_load : ℕ := 1

/-- Theorem stating that Sophie does 4 loads of laundry per week -/
theorem sophie_laundry_loads : 
  ∃ (loads_per_week : ℕ), 
    loads_per_week = 4 ∧ 
    (yearly_savings / box_cost : ℚ) * sheets_per_box = loads_per_week * weeks_per_year :=
sorry

end sophie_laundry_loads_l3351_335149


namespace quadratic_inequality_l3351_335131

def quadratic_function (x k : ℝ) : ℝ := -2 * x^2 + 4 * x + k

theorem quadratic_inequality (k : ℝ) :
  let x1 : ℝ := -0.99
  let x2 : ℝ := 0.98
  let x3 : ℝ := 0.99
  let y1 : ℝ := quadratic_function x1 k
  let y2 : ℝ := quadratic_function x2 k
  let y3 : ℝ := quadratic_function x3 k
  y1 < y2 ∧ y2 < y3 := by
sorry

end quadratic_inequality_l3351_335131


namespace wheel_speed_proof_l3351_335172

/-- Proves that the original speed of a wheel is 20 mph given specific conditions -/
theorem wheel_speed_proof (circumference : Real) (speed_increase : Real) (time_decrease : Real) :
  circumference = 50 / 5280 → -- circumference in miles
  speed_increase = 10 → -- speed increase in mph
  time_decrease = 1 / (2 * 3600) → -- time decrease in hours
  ∃ (r : Real),
    r > 0 ∧
    r * (50 * 3600 / (5280 * r)) = 50 / 5280 * 3600 ∧
    (r + speed_increase) * (50 * 3600 / (5280 * r) - time_decrease) = 50 / 5280 * 3600 ∧
    r = 20 :=
by sorry


end wheel_speed_proof_l3351_335172


namespace c_5_value_l3351_335150

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

theorem c_5_value (c : ℕ → ℝ) :
  geometric_sequence (λ n => Real.sqrt (c n)) ∧
  Real.sqrt (c 1) = 1 ∧
  Real.sqrt (c 2) = 2 * Real.sqrt (c 1) →
  c 5 = 256 := by
sorry

end c_5_value_l3351_335150


namespace bathroom_volume_l3351_335179

theorem bathroom_volume (length width height area volume : ℝ) : 
  length = 4 →
  area = 8 →
  height = 7 →
  area = length * width →
  volume = length * width * height →
  volume = 56 := by
sorry

end bathroom_volume_l3351_335179


namespace parallel_vectors_sum_l3351_335145

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

theorem parallel_vectors_sum (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → a.1 + b.1 = -2 ∧ a.2 + b.2 = -1 := by
  sorry

end parallel_vectors_sum_l3351_335145


namespace pioneer_assignment_l3351_335118

structure Pioneer where
  lastName : String
  firstName : String
  age : Nat

def Burov : Pioneer := sorry
def Gridnev : Pioneer := sorry
def Klimenko : Pioneer := sorry

axiom burov_not_kolya : Burov.firstName ≠ "Kolya"
axiom petya_school_start : ∃ p : Pioneer, p.firstName = "Petya" ∧ p.age = 12
axiom gridnev_grisha_older : Gridnev.age = (Klimenko.age + 1) ∧ Burov.age = (Klimenko.age + 1)

theorem pioneer_assignment :
  (Burov.firstName = "Grisha" ∧ Burov.age = 13) ∧
  (Gridnev.firstName = "Kolya" ∧ Gridnev.age = 13) ∧
  (Klimenko.firstName = "Petya" ∧ Klimenko.age = 12) :=
sorry

end pioneer_assignment_l3351_335118


namespace incorrect_conclusion_l3351_335155

theorem incorrect_conclusion (a b : ℝ) (h1 : a > b) (h2 : b > a + b) : 
  ¬(a * b > (a + b)^2) := by
sorry

end incorrect_conclusion_l3351_335155


namespace room002_is_selected_l3351_335148

/-- Represents a room number in the range [1, 60] -/
def RoomNumber := Fin 60

/-- The total number of examination rooms -/
def totalRooms : Nat := 60

/-- The number of rooms to be selected for inspection -/
def selectedRooms : Nat := 12

/-- The sample interval for systematic sampling -/
def sampleInterval : Nat := totalRooms / selectedRooms

/-- Predicate to check if a room is selected in the systematic sampling -/
def isSelected (room : RoomNumber) : Prop :=
  ∃ k : Nat, (room.val + 1) = k * sampleInterval + 2

/-- Theorem stating that room 002 is selected given the conditions -/
theorem room002_is_selected :
  isSelected ⟨1, by norm_num⟩ ∧ isSelected ⟨6, by norm_num⟩ := by
  sorry


end room002_is_selected_l3351_335148


namespace triangle_area_form_l3351_335189

/-- The radius of each circle -/
def r : ℝ := 44

/-- The side length of the equilateral triangle -/
noncomputable def s : ℝ := 2 * r * Real.sqrt 3

/-- The area of the equilateral triangle -/
noncomputable def area : ℝ := (s^2 * Real.sqrt 3) / 4

/-- Theorem stating the form of the area -/
theorem triangle_area_form :
  ∃ (a b : ℕ), area = Real.sqrt a + Real.sqrt b :=
sorry

end triangle_area_form_l3351_335189


namespace second_race_lead_l3351_335113

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race setup -/
structure Race where
  distance : ℝ
  sunny : Runner
  windy : Runner

theorem second_race_lead (h d : ℝ) (first_race second_race : Race) 
  (h_positive : h > 0)
  (d_positive : d > 0)
  (first_race_distance : first_race.distance = 2 * h)
  (second_race_distance : second_race.distance = 2 * h)
  (same_speeds : first_race.sunny.speed = second_race.sunny.speed ∧ 
                 first_race.windy.speed = second_race.windy.speed)
  (first_race_lead : first_race.sunny.speed * first_race.distance = 
                     first_race.windy.speed * (first_race.distance - 2 * d))
  (second_race_start : second_race.sunny.speed * (second_race.distance + 2 * d) = 
                       second_race.windy.speed * second_race.distance) :
  second_race.sunny.speed * second_race.distance - 
  second_race.windy.speed * second_race.distance = 2 * d^2 / h := by
  sorry

end second_race_lead_l3351_335113


namespace product_scaling_l3351_335128

theorem product_scaling (a b c : ℝ) (h : 14.97 * 46 = 688.62) : 
  1.497 * 4.6 = 6.8862 := by sorry

end product_scaling_l3351_335128


namespace intersection_A_complement_B_value_of_m_for_intersection_l3351_335157

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 ≤ 0}

-- Define set B with parameter m
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1: Intersection of A and complement of B when m = 3
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 3) = {x | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Theorem 2: Value of m when A ∩ B = {x | -1 ≤ x < 4}
theorem value_of_m_for_intersection :
  ∃ m : ℝ, A ∩ B m = {x | -1 ≤ x ∧ x < 4} ∧ m = 8 := by sorry

end intersection_A_complement_B_value_of_m_for_intersection_l3351_335157


namespace line_minimum_reciprocal_sum_l3351_335154

theorem line_minimum_reciprocal_sum (m n : ℝ) (h1 : m * n > 0) (h2 : m + n = 2) :
  ∀ (x y : ℝ), x * n + y * m = 2 → x = 1 ∧ y = 1 →
  (1 / m + 1 / n) ≥ 2 ∧ ∃ (m₀ n₀ : ℝ), m₀ * n₀ > 0 ∧ m₀ + n₀ = 2 ∧ 1 / m₀ + 1 / n₀ = 2 :=
by sorry

end line_minimum_reciprocal_sum_l3351_335154


namespace quadratic_equation_solution_l3351_335104

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 - 4*x + 4 = 0 ∧ x = 2 := by
  sorry

end quadratic_equation_solution_l3351_335104


namespace initial_number_of_persons_l3351_335197

theorem initial_number_of_persons (average_increase : ℝ) (weight_difference : ℝ) : 
  average_increase = 1.5 → weight_difference = 12 → 
  (average_increase * initial_persons = weight_difference) → initial_persons = 8 := by
  sorry

end initial_number_of_persons_l3351_335197


namespace min_redistributions_correct_l3351_335124

/-- Represents the redistribution process for a deck of 8 cards -/
def redistribute (deck : Vector ℕ 8) : Vector ℕ 8 :=
  sorry

/-- Checks if the deck is in its original order -/
def is_original_order (deck original : Vector ℕ 8) : Prop :=
  deck = original

/-- The minimum number of redistributions needed to restore the original order -/
def min_redistributions : ℕ := 3

/-- Theorem stating that the minimum number of redistributions to restore the original order is 3 -/
theorem min_redistributions_correct (original : Vector ℕ 8) :
  ∃ (n : ℕ), n = min_redistributions ∧
  ∀ (m : ℕ), m < n → ¬(is_original_order ((redistribute^[m]) original) original) ∧
  is_original_order ((redistribute^[n]) original) original :=
sorry

end min_redistributions_correct_l3351_335124


namespace smallest_m_for_distinct_roots_l3351_335102

theorem smallest_m_for_distinct_roots (m : ℤ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 4*x - m = 0 ∧ y^2 + 4*y - m = 0) → 
  (∀ k : ℤ, k < m → ¬∃ x y : ℝ, x ≠ y ∧ x^2 + 4*x - k = 0 ∧ y^2 + 4*y - k = 0) →
  m = -3 :=
by sorry

end smallest_m_for_distinct_roots_l3351_335102


namespace birthday_money_possibility_l3351_335162

theorem birthday_money_possibility (x y : ℕ) : ∃ (a : ℕ), 
  a < 10 ∧ 
  (x * y) % 10 = a ∧ 
  ((x + 1) * (y + 1)) % 10 = a ∧ 
  ((x + 2) * (y + 2)) % 10 = 0 := by
sorry

end birthday_money_possibility_l3351_335162


namespace payment_difference_l3351_335151

/-- Represents the pizza and its cost structure -/
structure Pizza :=
  (total_slices : ℕ)
  (plain_cost : ℚ)
  (anchovy_cost : ℚ)
  (mushroom_cost : ℚ)

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : ℚ :=
  p.plain_cost + p.anchovy_cost + p.mushroom_cost

/-- Calculates the number of slices Dave ate -/
def dave_slices (p : Pizza) : ℕ :=
  p.total_slices / 2 + p.total_slices / 4 + 1

/-- Calculates the number of slices Doug ate -/
def doug_slices (p : Pizza) : ℕ :=
  p.total_slices - dave_slices p

/-- Calculates Dave's payment -/
def dave_payment (p : Pizza) : ℚ :=
  total_cost p - (p.plain_cost / p.total_slices) * doug_slices p

/-- Calculates Doug's payment -/
def doug_payment (p : Pizza) : ℚ :=
  (p.plain_cost / p.total_slices) * doug_slices p

/-- The main theorem stating the difference in payments -/
theorem payment_difference (p : Pizza) 
  (h1 : p.total_slices = 8)
  (h2 : p.plain_cost = 8)
  (h3 : p.anchovy_cost = 2)
  (h4 : p.mushroom_cost = 1) :
  dave_payment p - doug_payment p = 9 := by
  sorry

end payment_difference_l3351_335151


namespace orange_distribution_difference_l3351_335192

/-- Calculates the difference in oranges per student between initial and final distribution --/
def orange_difference (total_oranges : ℕ) (bad_oranges : ℕ) (num_students : ℕ) : ℕ :=
  (total_oranges / num_students) - ((total_oranges - bad_oranges) / num_students)

theorem orange_distribution_difference :
  orange_difference 108 36 12 = 3 := by
  sorry

end orange_distribution_difference_l3351_335192


namespace imaginary_part_of_4_plus_3i_l3351_335174

theorem imaginary_part_of_4_plus_3i :
  Complex.im (4 + 3*Complex.I) = 3 := by
  sorry

end imaginary_part_of_4_plus_3i_l3351_335174


namespace part_one_part_two_l3351_335182

-- Define the sets A, B, and C
def A : Set ℝ := {1, 4, 7, 10}
def B (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 9}
def C : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

-- Part 1
theorem part_one :
  (A ∪ B 1 = {x | 1 ≤ x ∧ x ≤ 10}) ∧
  (A ∩ Cᶜ = {1, 7, 10}) := by sorry

-- Part 2
theorem part_two :
  ∀ m : ℝ, (B m ∩ C = C) → (-3 < m ∧ m < 3) := by sorry

end part_one_part_two_l3351_335182


namespace paint_cube_cost_l3351_335133

/-- The cost to paint a cube given paint price, coverage, and cube dimensions -/
theorem paint_cube_cost (paint_price : ℝ) (paint_coverage : ℝ) (cube_side : ℝ) : 
  paint_price = 36.5 →
  paint_coverage = 16 →
  cube_side = 8 →
  6 * cube_side^2 / paint_coverage * paint_price = 876 := by
sorry

end paint_cube_cost_l3351_335133


namespace divisibility_by_23_and_29_l3351_335196

theorem divisibility_by_23_and_29 (a b c : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) (hc : c ≤ 9) :
  ∃ (k m : ℕ), 200100 * a + 20010 * b + 2001 * c = 23 * k ∧ 200100 * a + 20010 * b + 2001 * c = 29 * m := by
  sorry

#check divisibility_by_23_and_29

end divisibility_by_23_and_29_l3351_335196


namespace fraction_addition_l3351_335117

theorem fraction_addition : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by
  sorry

end fraction_addition_l3351_335117


namespace largest_divisor_of_product_l3351_335139

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∃ k : ℕ, (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13) = 15 * k) ∧
  (∀ m : ℕ, m > 15 → ¬(∀ n : ℕ, Even n → n > 0 →
    ∃ k : ℕ, (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13) = m * k)) :=
by sorry

end largest_divisor_of_product_l3351_335139


namespace sum_of_roots_eq_nineteen_twelfths_l3351_335138

theorem sum_of_roots_eq_nineteen_twelfths :
  let f : ℝ → ℝ := λ x ↦ (4*x + 7) * (3*x - 10)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 19/12 :=
by sorry

end sum_of_roots_eq_nineteen_twelfths_l3351_335138


namespace statements_equivalent_l3351_335111

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_angles : angles 0 + angles 1 + angles 2 = π

-- Define an isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.angles 0 = t.angles 1 ∨ t.angles 1 = t.angles 2 ∨ t.angles 0 = t.angles 2

-- Define the three statements
def statement1 (t : Triangle) : Prop :=
  (∃ i j : Fin 3, i ≠ j ∧ t.angles i = t.angles j) → isIsosceles t

def statement2 (t : Triangle) : Prop :=
  ¬isIsosceles t → (∀ i j : Fin 3, i ≠ j → t.angles i ≠ t.angles j)

def statement3 (t : Triangle) : Prop :=
  (∃ i j : Fin 3, i ≠ j ∧ t.angles i = t.angles j) → isIsosceles t

-- Theorem: The three statements are logically equivalent
theorem statements_equivalent : ∀ t : Triangle,
  (statement1 t ↔ statement2 t) ∧ (statement2 t ↔ statement3 t) :=
sorry

end statements_equivalent_l3351_335111


namespace tony_squat_weight_l3351_335108

def curl_weight : ℕ := 90

def military_press_weight (curl : ℕ) : ℕ := 2 * curl

def squat_weight (military_press : ℕ) : ℕ := 5 * military_press

theorem tony_squat_weight : 
  squat_weight (military_press_weight curl_weight) = 900 := by
  sorry

end tony_squat_weight_l3351_335108


namespace average_speed_calculation_l3351_335191

/-- Given a run of 12 miles in 90 minutes, prove that the average speed is 8 miles per hour -/
theorem average_speed_calculation (distance : ℝ) (time_minutes : ℝ) (h1 : distance = 12) (h2 : time_minutes = 90) :
  distance / (time_minutes / 60) = 8 := by
sorry

end average_speed_calculation_l3351_335191


namespace log3_20_approximation_l3351_335140

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_5_approx : ℝ := 0.699

-- Define the target fraction
def target_fraction : ℚ := 33 / 12

-- Theorem statement
theorem log3_20_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |Real.log 20 / Real.log 3 - target_fraction| < ε :=
sorry

end log3_20_approximation_l3351_335140


namespace total_marbles_count_l3351_335136

theorem total_marbles_count (num_jars : ℕ) (marbles_per_jar : ℕ) : 
  num_jars = 16 →
  marbles_per_jar = 5 →
  (∃ (num_clay_pots : ℕ), num_jars = 2 * num_clay_pots) →
  (∃ (total_marbles : ℕ), 
    total_marbles = num_jars * marbles_per_jar + 
                    (num_jars / 2) * (3 * marbles_per_jar) ∧
    total_marbles = 200) :=
by
  sorry

#check total_marbles_count

end total_marbles_count_l3351_335136


namespace younger_brother_silver_fraction_l3351_335159

/-- The fraction of total silver received by the younger brother in a treasure division problem -/
theorem younger_brother_silver_fraction (x y : ℝ) 
  (h1 : x / 5 + y / 7 = 100)  -- Elder brother's share
  (h2 : x / 7 + (700 - x) / 7 = 100)  -- Younger brother's share
  : (700 - x) / (7 * y) = (y - (y - x / 5) / 2) / y := by
  sorry

end younger_brother_silver_fraction_l3351_335159


namespace geoffrey_remaining_money_l3351_335188

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial_amount : ℕ) (num_items : ℕ) (item_cost : ℕ) : ℕ :=
  initial_amount - num_items * item_cost

/-- Proves that Geoffrey has €20 left after his purchase -/
theorem geoffrey_remaining_money :
  remaining_money 125 3 35 = 20 := by
  sorry

end geoffrey_remaining_money_l3351_335188


namespace dividend_calculation_l3351_335181

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 19)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end dividend_calculation_l3351_335181


namespace cupcake_icing_time_l3351_335158

theorem cupcake_icing_time (total_batches : ℕ) (baking_time_per_batch : ℕ) (total_time : ℕ) :
  total_batches = 4 →
  baking_time_per_batch = 20 →
  total_time = 200 →
  (total_time - total_batches * baking_time_per_batch) / total_batches = 30 :=
by sorry

end cupcake_icing_time_l3351_335158


namespace anthony_total_pencils_l3351_335163

def initial_pencils : ℕ := 9
def gifted_pencils : ℕ := 56

theorem anthony_total_pencils : 
  initial_pencils + gifted_pencils = 65 := by
  sorry

end anthony_total_pencils_l3351_335163


namespace solve_for_y_l3351_335160

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end solve_for_y_l3351_335160


namespace square_difference_l3351_335141

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) :
  (x - y)^2 = 16 := by
  sorry

end square_difference_l3351_335141


namespace division_problem_l3351_335169

theorem division_problem (x : ℝ) 
  (h1 : ∃ k : ℝ, 3*k + 4*k + 5*k + 6*k = x)
  (h2 : ∃ m : ℝ, 3*m + 4*m + 6*m + 7*m = x)
  (h3 : ∃ (k m : ℝ), 6*m + 7*m = 5*k + 6*k + 1400) :
  x = 36000 := by
sorry

end division_problem_l3351_335169


namespace man_son_age_ratio_l3351_335165

/-- Given a man and his son, where the man is 34 years older than his son
    and the son's current age is 32, proves that the ratio of their ages
    in two years is 2:1. -/
theorem man_son_age_ratio :
  ∀ (son_age man_age : ℕ),
  son_age = 32 →
  man_age = son_age + 34 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end man_son_age_ratio_l3351_335165


namespace max_squares_on_8x8_board_l3351_335175

/-- Represents a checkerboard --/
structure Checkerboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a straight line on a checkerboard --/
structure Line :=
  (board : Checkerboard)

/-- Returns the maximum number of squares a line can pass through on a checkerboard --/
def max_squares_passed (l : Line) : Nat :=
  l.board.rows + l.board.cols - 1

/-- Theorem: The maximum number of squares a straight line can pass through on an 8x8 checkerboard is 15 --/
theorem max_squares_on_8x8_board :
  ∀ (l : Line), l.board = Checkerboard.mk 8 8 → max_squares_passed l = 15 := by
  sorry

end max_squares_on_8x8_board_l3351_335175


namespace sigma_odd_implies_perfect_square_l3351_335137

/-- The number of positive divisors of a natural number -/
def sigma (n : ℕ) : ℕ := sorry

/-- Theorem: If the number of positive divisors of a natural number is odd, then the number is a perfect square -/
theorem sigma_odd_implies_perfect_square (N : ℕ) : 
  Odd (sigma N) → ∃ m : ℕ, N = m ^ 2 := by
  sorry

end sigma_odd_implies_perfect_square_l3351_335137


namespace download_speed_scientific_notation_l3351_335122

/-- The network download speed of 5G in KB per second -/
def download_speed : ℕ := 1300000

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Convert a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : ScientificNotation := sorry

theorem download_speed_scientific_notation :
  to_scientific_notation download_speed = ScientificNotation.mk 1.3 6 := by sorry

end download_speed_scientific_notation_l3351_335122


namespace seventh_term_of_geometric_sequence_l3351_335168

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h1 : a 1 + a 2 = 3) 
  (h2 : a 2 + a 3 = 6) : 
  a 7 = 64 := by
sorry

end seventh_term_of_geometric_sequence_l3351_335168


namespace max_value_F_l3351_335134

theorem max_value_F (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → 
    |(a * x^2 + b * x + c) * (c * x^2 + b * x + a)| ≤ M ∧
    ∃ y : ℝ, y ∈ Set.Icc (-1) 1 ∧ 
      |(a * y^2 + b * y + c) * (c * y^2 + b * y + a)| = M) :=
by sorry

end max_value_F_l3351_335134


namespace wire_length_for_max_area_circular_sector_l3351_335198

/-- The length of wire needed to create a circular sector with maximum area --/
theorem wire_length_for_max_area_circular_sector (r : ℝ) (h : r = 4) :
  2 * π * r + 2 * r = 8 * π + 8 := by
  sorry

end wire_length_for_max_area_circular_sector_l3351_335198


namespace consecutive_integers_product_2720_sum_103_l3351_335101

theorem consecutive_integers_product_2720_sum_103 :
  ∀ n : ℕ, n > 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = 103 := by
  sorry

end consecutive_integers_product_2720_sum_103_l3351_335101


namespace tan_22_5_deg_sum_l3351_335171

theorem tan_22_5_deg_sum (a b c d : ℕ+) 
  (h1 : Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - b + (c : ℝ).sqrt - (d : ℝ).sqrt)
  (h2 : a ≥ b) (h3 : b ≥ c) (h4 : c ≥ d) : 
  a + b + c + d = 3 := by
  sorry

end tan_22_5_deg_sum_l3351_335171


namespace simplify_expression_l3351_335100

theorem simplify_expression (a : ℝ) : 5*a^2 - (a^2 - 2*(a^2 - 3*a)) = 6*a^2 - 6*a := by
  sorry

end simplify_expression_l3351_335100


namespace count_integers_with_conditions_l3351_335195

theorem count_integers_with_conditions : 
  ∃ (S : Finset ℤ), 
    (∀ n ∈ S, 150 < n ∧ n < 300 ∧ n % 7 = n % 9) ∧ 
    (∀ n, 150 < n → n < 300 → n % 7 = n % 9 → n ∈ S) ∧ 
    Finset.card S = 14 := by
  sorry

end count_integers_with_conditions_l3351_335195


namespace unique_integer_root_l3351_335123

def polynomial (x : ℤ) : ℤ := x^3 - 4*x^2 - 8*x + 24

theorem unique_integer_root : 
  (∀ x : ℤ, polynomial x = 0 ↔ x = 2) := by sorry

end unique_integer_root_l3351_335123


namespace star_diamond_relation_l3351_335183

theorem star_diamond_relation (star diamond : ℤ) 
  (h : 514 - star = 600 - diamond) : 
  star < diamond ∧ diamond - star = 86 := by
  sorry

end star_diamond_relation_l3351_335183


namespace arccos_cos_eleven_l3351_335129

theorem arccos_cos_eleven : 
  ∃! x : ℝ, x ∈ Set.Icc 0 π ∧ (x - 11) ∈ Set.range (λ n : ℤ => 2 * π * n) ∧ x = Real.arccos (Real.cos 11) := by
  sorry

end arccos_cos_eleven_l3351_335129


namespace min_folds_to_exceed_target_l3351_335185

def paper_thickness : ℝ := 0.1
def target_thickness : ℝ := 12

def thickness_after_folds (n : ℕ) : ℝ :=
  paper_thickness * (2 ^ n)

theorem min_folds_to_exceed_target : 
  ∀ n : ℕ, (thickness_after_folds n > target_thickness) ↔ (n ≥ 7) :=
sorry

end min_folds_to_exceed_target_l3351_335185


namespace computer_price_difference_l3351_335170

/-- Proof of the computer price difference problem -/
theorem computer_price_difference 
  (total_price : ℝ)
  (basic_price : ℝ)
  (printer_price : ℝ)
  (enhanced_price : ℝ)
  (h1 : total_price = 2500)
  (h2 : basic_price = 2000)
  (h3 : total_price = basic_price + printer_price)
  (h4 : printer_price = (1/6) * (enhanced_price + printer_price)) :
  enhanced_price - basic_price = 500 := by
  sorry

#check computer_price_difference

end computer_price_difference_l3351_335170


namespace lyka_initial_money_l3351_335112

/-- Calculates the initial amount of money Lyka has given the cost of a smartphone,
    the saving period in weeks, and the weekly saving rate. -/
def initial_money (smartphone_cost : ℕ) (saving_period : ℕ) (saving_rate : ℕ) : ℕ :=
  smartphone_cost - saving_period * saving_rate

/-- Proves that given a smartphone cost of $160, a saving period of 8 weeks,
    and a saving rate of $15 per week, the initial amount of money Lyka has is $40. -/
theorem lyka_initial_money :
  initial_money 160 8 15 = 40 := by
  sorry

end lyka_initial_money_l3351_335112


namespace fib_product_divisibility_l3351_335121

/-- Mersenne sequence property: for any two positive integers i and j, gcd(aᵢ, aⱼ) = a_{gcd(i,j)} -/
def is_mersenne_sequence (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i > 0 → j > 0 → Nat.gcd (a i) (a j) = a (Nat.gcd i j)

/-- Fibonacci sequence definition -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n+2 => fib (n+1) + fib n

/-- Product of first n terms of a sequence -/
def seq_product (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * a (i+1)) 1

/-- Main theorem: For Fibonacci sequence, product of k consecutive terms 
    is divisible by product of first k terms -/
theorem fib_product_divisibility (k : ℕ) (n : ℕ) : 
  k > 0 → is_mersenne_sequence fib → 
  (seq_product fib k) ∣ (List.range k).foldl (λ acc i => acc * fib (n+i)) 1 :=
sorry

end fib_product_divisibility_l3351_335121


namespace smallest_fraction_between_l3351_335126

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (2 : ℚ) / 3 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (2 : ℚ) / 3 → q ≤ q') →
  q - p = 3 := by
sorry

end smallest_fraction_between_l3351_335126


namespace product_of_cosines_equals_one_eighth_l3351_335120

theorem product_of_cosines_equals_one_eighth :
  (1 + Real.cos (π / 12)) * (1 + Real.cos (5 * π / 12)) *
  (1 + Real.cos (7 * π / 12)) * (1 + Real.cos (11 * π / 12)) = 1 / 8 := by
  sorry

end product_of_cosines_equals_one_eighth_l3351_335120


namespace quadratic_inequality_solution_l3351_335156

-- Define the quadratic function
def f (x : ℝ) := x^2 + 4*x - 5

-- Define the solution set
def solution_set : Set ℝ := {x | x < -5 ∨ x > 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
sorry

end quadratic_inequality_solution_l3351_335156


namespace pythagorean_theorem_3_4_5_l3351_335177

theorem pythagorean_theorem_3_4_5 : 
  ∀ (a b c : ℝ), 
    a = 3 → b = 4 → c^2 = a^2 + b^2 → c = 5 := by
  sorry

end pythagorean_theorem_3_4_5_l3351_335177


namespace slope_one_fourth_implies_y_six_l3351_335125

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is 1/4, then the y-coordinate of Q is 6. -/
theorem slope_one_fourth_implies_y_six (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = -3 →
  y₁ = 4 →
  x₂ = 5 →
  (y₂ - y₁) / (x₂ - x₁) = 1/4 →
  y₂ = 6 :=
by sorry

end slope_one_fourth_implies_y_six_l3351_335125


namespace marly_soup_bags_l3351_335130

/-- Calculates the number of bags needed to hold Marly's soup -/
def bags_needed (milk : ℚ) (chicken_stock_ratio : ℚ) (vegetables : ℚ) (bag_capacity : ℚ) : ℚ :=
  let total_volume := milk + (chicken_stock_ratio * milk) + vegetables
  total_volume / bag_capacity

/-- Proves that Marly needs 3 bags for his soup -/
theorem marly_soup_bags :
  bags_needed 2 3 1 3 = 3 := by
  sorry

end marly_soup_bags_l3351_335130


namespace denominator_value_l3351_335178

theorem denominator_value (p q x : ℚ) : 
  p / q = 4 / 5 → 
  4 / 7 + (2 * q - p) / x = 1 → 
  x = 7 := by
sorry

end denominator_value_l3351_335178


namespace gcd_factorial_eight_ten_l3351_335161

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_eight_ten :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end gcd_factorial_eight_ten_l3351_335161


namespace tan_8100_degrees_l3351_335110

theorem tan_8100_degrees : Real.tan (8100 * π / 180) = 0 := by
  sorry

end tan_8100_degrees_l3351_335110


namespace area_difference_zero_l3351_335176

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  contains_origin : (center.1)^2 + (center.2)^2 < radius^2
  radius : ℝ

-- Define the areas S+ and S-
def S_plus (c : Circle) : ℝ := sorry
def S_minus (c : Circle) : ℝ := sorry

-- Theorem statement
theorem area_difference_zero (c : Circle) : S_plus c - S_minus c = 0 := by sorry

end area_difference_zero_l3351_335176


namespace basketball_score_proof_l3351_335152

theorem basketball_score_proof (two_point_shots three_point_shots free_throws : ℕ) : 
  (3 * three_point_shots = 2 * two_point_shots) →
  (free_throws = 2 * two_point_shots) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 72) →
  free_throws = 24 := by
  sorry

#check basketball_score_proof

end basketball_score_proof_l3351_335152


namespace heechul_most_books_l3351_335164

/-- The number of books each person has -/
structure BookCollection where
  heejin : ℕ
  heechul : ℕ
  dongkyun : ℕ

/-- Conditions of the book collection -/
def valid_collection (bc : BookCollection) : Prop :=
  bc.heechul = bc.heejin + 2 ∧ bc.dongkyun < bc.heejin

/-- Heechul has the most books -/
def heechul_has_most (bc : BookCollection) : Prop :=
  bc.heechul > bc.heejin ∧ bc.heechul > bc.dongkyun

/-- Theorem: If the collection is valid, then Heechul has the most books -/
theorem heechul_most_books (bc : BookCollection) :
  valid_collection bc → heechul_has_most bc := by
  sorry

end heechul_most_books_l3351_335164


namespace geometric_sequence_common_ratio_l3351_335190

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_a2 : a 2 = 3) 
  (h_a6 : a 6 = 48) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end geometric_sequence_common_ratio_l3351_335190


namespace fraction_of_apples_sold_l3351_335114

/-- Proves that the fraction of apples sold is 1/2 given the initial quantities and conditions --/
theorem fraction_of_apples_sold
  (initial_oranges : ℕ)
  (initial_apples : ℕ)
  (orange_fraction_sold : ℚ)
  (total_fruits_left : ℕ)
  (h1 : initial_oranges = 40)
  (h2 : initial_apples = 70)
  (h3 : orange_fraction_sold = 1/4)
  (h4 : total_fruits_left = 65)
  : (initial_apples - (total_fruits_left - (initial_oranges - initial_oranges * orange_fraction_sold))) / initial_apples = 1/2 := by
  sorry

end fraction_of_apples_sold_l3351_335114


namespace analysis_time_proof_l3351_335116

/-- The number of bones in a human body -/
def num_bones : ℕ := 206

/-- The time spent analyzing each bone (in hours) -/
def time_per_bone : ℕ := 1

/-- The total time needed to analyze all bones in a human body -/
def total_analysis_time : ℕ := num_bones * time_per_bone

theorem analysis_time_proof : total_analysis_time = 206 := by
  sorry

end analysis_time_proof_l3351_335116


namespace smallest_four_digit_sum_27_l3351_335132

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_sum_27 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 27 → 1899 ≤ n :=
by sorry

end smallest_four_digit_sum_27_l3351_335132


namespace geometric_series_common_ratio_l3351_335146

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -14/27
  let a₃ : ℚ := 56/243
  let r : ℚ := -16/27
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by sorry

end geometric_series_common_ratio_l3351_335146


namespace max_score_in_twenty_over_match_l3351_335106

/-- Represents the number of overs in the cricket match -/
def overs : ℕ := 20

/-- Represents the number of balls in an over -/
def balls_per_over : ℕ := 6

/-- Represents the maximum runs that can be scored on a single ball -/
def max_runs_per_ball : ℕ := 6

/-- Calculates the maximum runs a batsman can score in a perfect scenario -/
def max_batsman_score : ℕ := overs * balls_per_over * max_runs_per_ball

theorem max_score_in_twenty_over_match :
  max_batsman_score = 720 :=
by sorry

end max_score_in_twenty_over_match_l3351_335106


namespace trigonometric_identity_l3351_335153

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : 0 < a) (h' : 0 < b) 
  (h_identity : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) :
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 := by
  sorry

end trigonometric_identity_l3351_335153
