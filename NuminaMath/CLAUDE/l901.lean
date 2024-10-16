import Mathlib

namespace NUMINAMATH_CALUDE_ticket_price_is_25_l901_90190

-- Define the number of attendees for the first show
def first_show_attendees : ℕ := 200

-- Define the number of attendees for the second show
def second_show_attendees : ℕ := 3 * first_show_attendees

-- Define the total revenue
def total_revenue : ℕ := 20000

-- Define the ticket price
def ticket_price : ℚ := total_revenue / (first_show_attendees + second_show_attendees)

-- Theorem statement
theorem ticket_price_is_25 : ticket_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_is_25_l901_90190


namespace NUMINAMATH_CALUDE_min_monochromatic_triangles_l901_90127

/-- A coloring of edges in a complete graph on 2k vertices. -/
def Coloring (k : ℕ) := Fin (2*k) → Fin (2*k) → Bool

/-- The number of monochromatic triangles in a coloring. -/
def monochromaticTriangles (k : ℕ) (c : Coloring k) : ℕ := sorry

/-- The statement of the problem. -/
theorem min_monochromatic_triangles (k : ℕ) (h : k ≥ 3) :
  ∃ (c : Coloring k), monochromaticTriangles k c = k * (k - 1) * (k - 2) / 3 ∧
  ∀ (c' : Coloring k), monochromaticTriangles k c' ≥ k * (k - 1) * (k - 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_monochromatic_triangles_l901_90127


namespace NUMINAMATH_CALUDE_constant_function_derivative_l901_90151

-- Define the constant function f(x) = 0
def f : ℝ → ℝ := λ x ↦ 0

-- State the theorem
theorem constant_function_derivative :
  ∀ x : ℝ, deriv f x = 0 := by sorry

end NUMINAMATH_CALUDE_constant_function_derivative_l901_90151


namespace NUMINAMATH_CALUDE_subset_condition_l901_90195

/-- The set A defined by the given condition -/
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (a + 1)) < 0}

/-- The set B defined by the given condition -/
def B (a : ℝ) : Set ℝ := {x | (x - 2*a) / (x - (a^2 + 1)) < 0}

/-- The theorem stating the relationship between a and the subset property -/
theorem subset_condition (a : ℝ) : 
  B a ⊆ A a ↔ a ∈ Set.Icc (-1/2) (-1/2) ∪ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l901_90195


namespace NUMINAMATH_CALUDE_cos_fourth_power_identity_l901_90184

theorem cos_fourth_power_identity (θ : ℝ) : 
  (Real.cos θ)^4 = (1/8) * Real.cos (4*θ) + (1/2) * Real.cos (2*θ) + 0 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_cos_fourth_power_identity_l901_90184


namespace NUMINAMATH_CALUDE_train_passes_jogger_train_passes_jogger_time_l901_90163

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed : Real) (train_speed : Real) 
  (jogger_lead : Real) (train_length : Real) : Real :=
  let jogger_speed_ms := jogger_speed * (1000 / 3600)
  let train_speed_ms := train_speed * (1000 / 3600)
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := jogger_lead + train_length
  total_distance / relative_speed

/-- The time for the train to pass the jogger is 24 seconds -/
theorem train_passes_jogger_time :
  train_passes_jogger 9 45 120 120 = 24 := by
  sorry

end NUMINAMATH_CALUDE_train_passes_jogger_train_passes_jogger_time_l901_90163


namespace NUMINAMATH_CALUDE_rabbit_speed_l901_90192

def rabbit_speed_equation (x : ℝ) : Prop :=
  2 * (2 * x + 4) = 188

theorem rabbit_speed : ∃ (x : ℝ), rabbit_speed_equation x ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_speed_l901_90192


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l901_90158

theorem square_root_of_sixteen (x : ℝ) : x^2 = 16 ↔ x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l901_90158


namespace NUMINAMATH_CALUDE_sum_equality_l901_90189

def sum_ascending (n : ℕ) : ℕ := (n * (n + 1)) / 2

def sum_descending (n : ℕ) : ℕ := 
  if n = 0 then 0 else n + sum_descending (n - 1)

theorem sum_equality : 
  sum_ascending 1000 = sum_descending 1000 :=
by sorry

end NUMINAMATH_CALUDE_sum_equality_l901_90189


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l901_90153

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, 
    x₁ = (11 + Real.sqrt 13) / 6 ∧ 
    x₂ = (11 - Real.sqrt 13) / 6 ∧ 
    (x₁ - 2) * (3 * x₁ - 5) = 1 ∧ 
    (x₂ - 2) * (3 * x₂ - 5) = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l901_90153


namespace NUMINAMATH_CALUDE_raisin_difference_is_twenty_l901_90106

/-- The number of raisin cookies Helen baked yesterday -/
def yesterday_raisin : ℕ := 300

/-- The number of raisin cookies Helen baked today -/
def today_raisin : ℕ := 280

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_chocolate : ℕ := 519

/-- The number of chocolate chip cookies Helen baked today -/
def today_chocolate : ℕ := 359

/-- The difference in raisin cookies baked between yesterday and today -/
def raisin_difference : ℕ := yesterday_raisin - today_raisin

theorem raisin_difference_is_twenty : raisin_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_raisin_difference_is_twenty_l901_90106


namespace NUMINAMATH_CALUDE_min_faces_two_dice_l901_90188

structure Dice where
  faces : ℕ
  min_faces : ℕ
  distinct_numbering : faces ≥ min_faces

def probability_sum (a b : Dice) (sum : ℕ) : ℚ :=
  (Finset.filter (fun (x, y) => x + y = sum) (Finset.product (Finset.range a.faces) (Finset.range b.faces))).card /
  (a.faces * b.faces : ℚ)

theorem min_faces_two_dice (a b : Dice) : 
  a.min_faces = 7 → 
  b.min_faces = 5 → 
  probability_sum a b 13 = 2 * probability_sum a b 8 →
  probability_sum a b 16 = 1/20 →
  a.faces + b.faces ≥ 24 ∧ 
  ∀ (a' b' : Dice), a'.faces + b'.faces < 24 → 
    (a'.min_faces = 7 ∧ b'.min_faces = 5 ∧ 
     probability_sum a' b' 13 = 2 * probability_sum a' b' 8 ∧
     probability_sum a' b' 16 = 1/20) → False :=
by sorry

end NUMINAMATH_CALUDE_min_faces_two_dice_l901_90188


namespace NUMINAMATH_CALUDE_unique_valid_multiplication_l901_90161

def is_valid_multiplication (a b : Nat) : Prop :=
  a % 10 = 5 ∧
  b % 10 = 5 ∧
  (a * (b / 10 % 10)) % 100 = 25 ∧
  (a / 10 % 10) % 2 = 0 ∧
  b / 10 % 10 < 3 ∧
  1000 ≤ a * b ∧ a * b < 10000

theorem unique_valid_multiplication :
  ∀ a b : Nat, is_valid_multiplication a b → (a = 365 ∧ b = 25) :=
sorry

end NUMINAMATH_CALUDE_unique_valid_multiplication_l901_90161


namespace NUMINAMATH_CALUDE_remainder_problem_l901_90110

theorem remainder_problem (n : ℤ) (h : n % 7 = 2) : (4 * n + 5) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l901_90110


namespace NUMINAMATH_CALUDE_bethany_portraits_l901_90144

theorem bethany_portraits (total_paintings : ℕ) (still_life_ratio : ℕ) : 
  total_paintings = 200 → still_life_ratio = 6 →
  ∃ (portraits : ℕ), portraits = 28 ∧ 
    portraits * (still_life_ratio + 1) = total_paintings :=
by sorry

end NUMINAMATH_CALUDE_bethany_portraits_l901_90144


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l901_90120

/-- Proves the time taken for a train to cross a bridge given its length, speed, and time to pass a fixed point on the bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (signal_post_time : ℝ) 
  (bridge_fixed_point_time : ℝ) 
  (h1 : train_length = 600) 
  (h2 : signal_post_time = 40) 
  (h3 : bridge_fixed_point_time = 1200) :
  let train_speed := train_length / signal_post_time
  let bridge_length := train_speed * bridge_fixed_point_time - train_length
  let total_distance := bridge_length + train_length
  total_distance / train_speed = 1240 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l901_90120


namespace NUMINAMATH_CALUDE_line_perpendicular_sufficient_condition_l901_90124

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perp : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpToPlane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_sufficient_condition
  (l m : Line) (α β : Plane)
  (h1 : intersect α β = l)
  (h2 : subset m β)
  (h3 : perpToPlane m α) :
  perp l m :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_sufficient_condition_l901_90124


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l901_90172

theorem quadratic_root_problem (a : ℝ) : 
  ((a - 1) * 1^2 - a * 1 + a^2 = 0) → (a ≠ 1) → (a = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l901_90172


namespace NUMINAMATH_CALUDE_original_list_size_l901_90179

/-- Given a list of integers, if appending 25 increases the mean by 3,
    and then appending -4 decreases the mean by 1.5,
    prove that the original list contained 4 integers. -/
theorem original_list_size (l : List Int) : 
  (((l.sum + 25) / (l.length + 1) : ℚ) = (l.sum / l.length : ℚ) + 3) →
  (((l.sum + 21) / (l.length + 2) : ℚ) = (l.sum / l.length : ℚ) + 1.5) →
  l.length = 4 := by
sorry


end NUMINAMATH_CALUDE_original_list_size_l901_90179


namespace NUMINAMATH_CALUDE_equation_solution_l901_90197

theorem equation_solution : ∃! x : ℝ, 4 * x - 8 + 3 * x = 12 + 5 * x ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l901_90197


namespace NUMINAMATH_CALUDE_dice_surface_sum_in_possible_sums_l901_90199

/-- The number of dice in the arrangement -/
def num_dice : ℕ := 2012

/-- The sum of points on all faces of a standard six-sided die -/
def die_sum : ℕ := 21

/-- The sum of points on opposite faces of a die -/
def opposite_faces_sum : ℕ := 7

/-- The set of possible sums of points on the surface -/
def possible_sums : Set ℕ := {28177, 28179, 28181, 28183, 28185, 28187}

/-- Theorem: The sum of points on the surface of the arranged dice is in the set of possible sums -/
theorem dice_surface_sum_in_possible_sums :
  ∃ (x : ℕ), x ∈ possible_sums ∧
  ∃ (end_face_sum : ℕ), end_face_sum ≥ 1 ∧ end_face_sum ≤ 6 ∧
  x = num_dice * die_sum - (num_dice - 1) * opposite_faces_sum + 2 * end_face_sum :=
by
  sorry

end NUMINAMATH_CALUDE_dice_surface_sum_in_possible_sums_l901_90199


namespace NUMINAMATH_CALUDE_bookstore_shipment_l901_90133

/-- Proves that the total number of books in a shipment is 240, given that 25% are displayed
    in the front and the remaining 180 books are in the storage room. -/
theorem bookstore_shipment (displayed_percent : ℚ) (storage_count : ℕ) : ℕ :=
  let total_books : ℕ := 240
  have h1 : displayed_percent = 25 / 100 := by sorry
  have h2 : storage_count = 180 := by sorry
  have h3 : (1 - displayed_percent) * total_books = storage_count := by sorry
  total_books

#check bookstore_shipment

end NUMINAMATH_CALUDE_bookstore_shipment_l901_90133


namespace NUMINAMATH_CALUDE_waysToChooseIsCorrect_l901_90155

/-- The number of ways to choose a president and a 3-person committee from a group of 10 people -/
def waysToChoose : ℕ :=
  let totalPeople : ℕ := 10
  let committeeSize : ℕ := 3
  totalPeople * Nat.choose (totalPeople - 1) committeeSize

/-- Theorem stating that the number of ways to choose a president and a 3-person committee
    from a group of 10 people is 840 -/
theorem waysToChooseIsCorrect : waysToChoose = 840 := by
  sorry

end NUMINAMATH_CALUDE_waysToChooseIsCorrect_l901_90155


namespace NUMINAMATH_CALUDE_probability_two_hearts_one_spade_l901_90154

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of hearts in a standard deck -/
def numberOfHearts : ℕ := 13

/-- The number of spades in a standard deck -/
def numberOfSpades : ℕ := 13

/-- The probability of drawing two hearts followed by a spade from a standard 52-card deck -/
theorem probability_two_hearts_one_spade :
  (numberOfHearts * (numberOfHearts - 1) * numberOfSpades : ℚ) / 
  (standardDeckSize * (standardDeckSize - 1) * (standardDeckSize - 2)) = 78 / 5115 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_hearts_one_spade_l901_90154


namespace NUMINAMATH_CALUDE_length_of_segment_AB_l901_90177

/-- Given two perpendicular lines and a point P, prove the length of AB --/
theorem length_of_segment_AB (a : ℝ) : 
  ∃ (A B : ℝ × ℝ),
    (2 * A.1 - A.2 = 0) ∧ 
    (B.1 + a * B.2 = 0) ∧
    ((0 : ℝ) = (A.1 + B.1) / 2) ∧
    ((10 / a) = (A.2 + B.2) / 2) ∧
    (2 * a = -1) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_length_of_segment_AB_l901_90177


namespace NUMINAMATH_CALUDE_cycle_price_proof_l901_90164

/-- Proves that a cycle sold at a 5% loss for 1330 had an original price of 1400 -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1330)
  (h2 : loss_percentage = 5) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_proof_l901_90164


namespace NUMINAMATH_CALUDE_constant_zero_is_arithmetic_not_geometric_l901_90152

def constant_zero_sequence : ℕ → ℝ := fun _ ↦ 0

theorem constant_zero_is_arithmetic_not_geometric :
  (∃ d : ℝ, ∀ n : ℕ, constant_zero_sequence (n + 1) = constant_zero_sequence n + d) ∧
  (¬ ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, constant_zero_sequence (n + 1) = constant_zero_sequence n * r) :=
by sorry

end NUMINAMATH_CALUDE_constant_zero_is_arithmetic_not_geometric_l901_90152


namespace NUMINAMATH_CALUDE_min_value_theorem_l901_90186

theorem min_value_theorem (x y : ℝ) (h : Real.log x + Real.log y = 1) :
  (2 / x + 5 / y) ≥ 2 ∧ ∃ (x₀ y₀ : ℝ), Real.log x₀ + Real.log y₀ = 1 ∧ 2 / x₀ + 5 / y₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l901_90186


namespace NUMINAMATH_CALUDE_largest_prime_factor_l901_90171

theorem largest_prime_factor (n : ℕ) : 
  (∃ p : ℕ, Prime p ∧ p ∣ (17^4 + 3 * 17^2 + 1 - 16^4) ∧ 
    ∀ q : ℕ, Prime q → q ∣ (17^4 + 3 * 17^2 + 1 - 16^4) → q ≤ p) → 
  (∃ p : ℕ, p = 17 ∧ Prime p ∧ p ∣ (17^4 + 3 * 17^2 + 1 - 16^4) ∧ 
    ∀ q : ℕ, Prime q → q ∣ (17^4 + 3 * 17^2 + 1 - 16^4) → q ≤ p) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l901_90171


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l901_90114

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l901_90114


namespace NUMINAMATH_CALUDE_min_value_x3_l901_90181

theorem min_value_x3 (x₁ x₂ x₃ : ℝ) 
  (eq1 : x₁ + (1/2) * x₂ + (1/3) * x₃ = 1)
  (eq2 : x₁^2 + (1/2) * x₂^2 + (1/3) * x₃^2 = 3) :
  x₃ ≥ -21/11 ∧ ∃ (x₁' x₂' x₃' : ℝ), 
    x₁' + (1/2) * x₂' + (1/3) * x₃' = 1 ∧
    x₁'^2 + (1/2) * x₂'^2 + (1/3) * x₃'^2 = 3 ∧
    x₃' = -21/11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x3_l901_90181


namespace NUMINAMATH_CALUDE_circle_center_transformation_l901_90149

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 - d)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (-2, 6)
  let reflected := reflect_y initial_center
  let final_position := translate_down reflected 8
  final_position = (2, -2) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l901_90149


namespace NUMINAMATH_CALUDE_monitor_pixels_l901_90147

/-- Calculates the total number of pixels on a monitor given its dimensions and DPI. -/
theorem monitor_pixels 
  (width : ℕ) 
  (height : ℕ) 
  (dpi : ℕ) 
  (h1 : width = 21) 
  (h2 : height = 12) 
  (h3 : dpi = 100) : 
  width * dpi * (height * dpi) = 2520000 := by
  sorry

#check monitor_pixels

end NUMINAMATH_CALUDE_monitor_pixels_l901_90147


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l901_90185

def selling_price : ℝ := 750
def cost : ℝ := 555.56

theorem profit_percentage_calculation :
  let profit := selling_price - cost
  let percentage := (profit / cost) * 100
  ∃ ε > 0, abs (percentage - 34.99) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l901_90185


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l901_90115

/-- Given points (2, y₁) and (-2, y₂) on the graph of y = ax² + bx + d, where y₁ - y₂ = -8, prove that b = -2 -/
theorem quadratic_coefficient (a d y₁ y₂ : ℝ) : 
  y₁ = 4 * a + 2 * b + d →
  y₂ = 4 * a - 2 * b + d →
  y₁ - y₂ = -8 →
  b = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l901_90115


namespace NUMINAMATH_CALUDE_work_speed_ratio_l901_90119

-- Define the work speeds
def A_work_speed : ℚ := 1 / 18
def B_work_speed : ℚ := 1 / 36

-- Define the combined work speed
def combined_work_speed : ℚ := 1 / 12

-- Theorem statement
theorem work_speed_ratio :
  (A_work_speed + B_work_speed = combined_work_speed) →
  (A_work_speed / B_work_speed = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_work_speed_ratio_l901_90119


namespace NUMINAMATH_CALUDE_gcd_problem_l901_90116

theorem gcd_problem (n : ℕ) (a : Fin n → ℕ+) 
  (h_odd : Odd n) 
  (h_n_gt_1 : n > 1) 
  (h_gcd_1 : Nat.gcd (Finset.univ.prod (fun i => (a i))) = 1) :
  let d := Nat.gcd (Finset.univ.prod (fun i => (a i).val^n + (Finset.univ.prod (fun j => (a j)))))
  d = 1 ∨ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l901_90116


namespace NUMINAMATH_CALUDE_water_tank_problem_l901_90156

/-- Represents the time (in minutes) it takes for pipe A to fill the tank -/
def fill_time : ℚ := 15

/-- Represents the time (in minutes) it takes for pipe B to empty the tank -/
def empty_time : ℚ := 6

/-- Represents the time (in minutes) it takes to empty or fill the tank completely with both pipes open -/
def both_pipes_time : ℚ := 2

/-- Represents the fraction of the tank that is currently full -/
def current_fill : ℚ := 4/5

theorem water_tank_problem :
  (1 / fill_time - 1 / empty_time) * both_pipes_time = 1 - current_fill :=
by sorry

end NUMINAMATH_CALUDE_water_tank_problem_l901_90156


namespace NUMINAMATH_CALUDE_new_person_age_l901_90113

/-- Given a group of people with an initial average age and size, 
    calculate the age of a new person that changes the average to a new value. -/
theorem new_person_age (n : ℕ) (initial_avg new_avg : ℚ) : 
  n = 17 → 
  initial_avg = 14 → 
  new_avg = 15 → 
  (n : ℚ) * initial_avg + (new_avg * ((n : ℚ) + 1) - (n : ℚ) * initial_avg) = 32 := by
  sorry

#check new_person_age

end NUMINAMATH_CALUDE_new_person_age_l901_90113


namespace NUMINAMATH_CALUDE_equation_solutions_l901_90100

theorem equation_solutions : 
  let f (x : ℝ) := 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5))
  ∀ x : ℝ, f x = 1/8 ↔ x = 7 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l901_90100


namespace NUMINAMATH_CALUDE_election_votes_proof_l901_90159

theorem election_votes_proof (total_votes : ℕ) (invalid_percentage : ℚ) (winner_percentage : ℚ) :
  total_votes = 7000 →
  invalid_percentage = 1/5 →
  winner_percentage = 11/20 →
  let valid_votes := total_votes - (invalid_percentage * total_votes).num
  let winner_votes := (winner_percentage * valid_votes).num
  valid_votes - winner_votes = 2520 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_proof_l901_90159


namespace NUMINAMATH_CALUDE_solve_system_l901_90169

theorem solve_system (x y : ℝ) 
  (eq1 : 2 * x = 3 * x - 25)
  (eq2 : x + y = 50) : 
  x = 25 ∧ y = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l901_90169


namespace NUMINAMATH_CALUDE_office_work_distribution_l901_90183

theorem office_work_distribution (P : ℕ) : P > 0 → (6 / 7 : ℚ) * P * (6 / 5 : ℚ) = P → P ≥ 35 := by
  sorry

end NUMINAMATH_CALUDE_office_work_distribution_l901_90183


namespace NUMINAMATH_CALUDE_system_solution_condition_l901_90157

theorem system_solution_condition (a : ℕ+) (A B : ℝ) :
  (∃ x y z : ℕ+, 
    x^2 + y^2 + z^2 = (B * (a : ℝ))^2 ∧
    x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = 
      (1/4) * (2*A + B) * (B * (a : ℝ))^4) ↔
  B = 2 * A := by
sorry

end NUMINAMATH_CALUDE_system_solution_condition_l901_90157


namespace NUMINAMATH_CALUDE_unique_solution_value_l901_90141

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant (b^2 - 4ac) must be zero -/
def has_unique_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 3x^2 - 7x + k = 0 -/
def quadratic_equation (k : ℝ) (x : ℝ) : Prop :=
  3*x^2 - 7*x + k = 0

theorem unique_solution_value (k : ℝ) :
  (∃! x, quadratic_equation k x) ↔ k = 49/12 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_value_l901_90141


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l901_90140

theorem polynomial_division_quotient :
  ∀ x : ℝ, x ≠ 1 →
  (x^6 + 6) = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l901_90140


namespace NUMINAMATH_CALUDE_existence_of_special_number_l901_90136

theorem existence_of_special_number (P : Finset Nat) (h_prime : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : Nat,
    (∀ p ∈ P, ∃ a b : Nat, x = a^p + b^p) ∧
    (∀ p : Nat, Nat.Prime p → p ∉ P → ¬∃ a b : Nat, x = a^p + b^p) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l901_90136


namespace NUMINAMATH_CALUDE_unique_divisible_by_18_l901_90126

def is_divisible_by_18 (n : ℕ) : Prop := n % 18 = 0

def four_digit_number (x : ℕ) : ℕ := x * 1000 + 520 + x

theorem unique_divisible_by_18 :
  ∃! x : ℕ, x < 10 ∧ is_divisible_by_18 (four_digit_number x) :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_18_l901_90126


namespace NUMINAMATH_CALUDE_power_three_315_mod_11_l901_90174

theorem power_three_315_mod_11 : 3^315 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_315_mod_11_l901_90174


namespace NUMINAMATH_CALUDE_min_real_roots_l901_90168

/-- A polynomial of degree 2010 with real coefficients -/
def RealPolynomial2010 : Type := Polynomial ℝ

/-- The roots of a polynomial -/
def roots (p : RealPolynomial2010) : Multiset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinctAbsValues (p : RealPolynomial2010) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def realRootCount (p : RealPolynomial2010) : ℕ := sorry

/-- The degree of the polynomial -/
def degree (p : RealPolynomial2010) : ℕ := 2010

theorem min_real_roots (g : RealPolynomial2010) 
  (h1 : degree g = 2010)
  (h2 : distinctAbsValues g = 1006) : 
  realRootCount g ≥ 6 := sorry

end NUMINAMATH_CALUDE_min_real_roots_l901_90168


namespace NUMINAMATH_CALUDE_dogGroupings_eq_2520_l901_90111

/-- The number of ways to divide 12 dogs into groups of 4, 6, and 2,
    with Rover in the 4-dog group and Spot in the 6-dog group. -/
def dogGroupings : ℕ :=
  (Nat.choose 10 3) * (Nat.choose 7 5)

/-- Theorem stating that the number of ways to divide the dogs is 2520. -/
theorem dogGroupings_eq_2520 : dogGroupings = 2520 := by
  sorry

end NUMINAMATH_CALUDE_dogGroupings_eq_2520_l901_90111


namespace NUMINAMATH_CALUDE_base_10_500_equals_base_6_2152_l901_90180

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 6 to a natural number -/
def fromBase6 (digits : List ℕ) : ℕ :=
  sorry

theorem base_10_500_equals_base_6_2152 :
  toBase6 500 = [2, 1, 5, 2] ∧ fromBase6 [2, 1, 5, 2] = 500 :=
sorry

end NUMINAMATH_CALUDE_base_10_500_equals_base_6_2152_l901_90180


namespace NUMINAMATH_CALUDE_peach_probability_l901_90196

/-- A fruit type in the basket -/
inductive Fruit
| apple
| pear
| peach

/-- The number of fruits in the basket -/
def basket : Fruit → ℕ
| Fruit.apple => 5
| Fruit.pear => 3
| Fruit.peach => 2

/-- The total number of fruits in the basket -/
def total_fruits : ℕ := basket Fruit.apple + basket Fruit.pear + basket Fruit.peach

/-- The probability of picking a specific fruit -/
def prob_pick (f : Fruit) : ℚ := basket f / total_fruits

theorem peach_probability :
  prob_pick Fruit.peach = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_peach_probability_l901_90196


namespace NUMINAMATH_CALUDE_secret_society_friendships_l901_90139

/-- Represents a member of the secret society -/
structure Member where
  balance : Int

/-- Represents the secret society -/
structure SecretSociety where
  members : Finset Member
  friendships : Finset (Member × Member)
  
/-- A function that represents giving one dollar to all friends -/
def giveDollarToFriends (s : SecretSociety) (m : Member) : SecretSociety :=
  sorry

/-- A predicate that checks if money can be arbitrarily redistributed -/
def canRedistributeArbitrarily (s : SecretSociety) : Prop :=
  sorry

theorem secret_society_friendships 
  (s : SecretSociety) 
  (h1 : s.members.card = 2011) 
  (h2 : canRedistributeArbitrarily s) : 
  s.friendships.card = 2010 :=
sorry

end NUMINAMATH_CALUDE_secret_society_friendships_l901_90139


namespace NUMINAMATH_CALUDE_ceiling_plus_self_eq_150_l901_90145

theorem ceiling_plus_self_eq_150 :
  ∃ y : ℝ, ⌈y⌉ + y = 150 ∧ y = 75 := by sorry

end NUMINAMATH_CALUDE_ceiling_plus_self_eq_150_l901_90145


namespace NUMINAMATH_CALUDE_airport_distance_l901_90132

/-- Proves that the distance to the airport is 315 miles given the problem conditions -/
theorem airport_distance : 
  ∀ (d : ℝ) (t : ℝ),
  (d = 45 * (t + 1.5)) →  -- If continued at initial speed, arriving on time
  (d - 45 = 60 * (t - 1)) →  -- Adjusted speed for remaining journey, arriving 1 hour early
  d = 315 := by sorry

end NUMINAMATH_CALUDE_airport_distance_l901_90132


namespace NUMINAMATH_CALUDE_square_cut_into_three_rectangles_l901_90167

theorem square_cut_into_three_rectangles :
  ∀ (a b c d e : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  a + b = 36 ∧ c + d = 36 ∧ a + c = 36 →
  a * e = b * (36 - e) ∧ c * e = d * (36 - e) →
  (∃ x y : ℝ, (x = a ∨ x = b) ∧ (y = c ∨ y = d) ∧ x + y = 36) →
  36 + e = 60 :=
by sorry

end NUMINAMATH_CALUDE_square_cut_into_three_rectangles_l901_90167


namespace NUMINAMATH_CALUDE_zucchini_weight_l901_90193

/-- Proves that the weight of zucchini installed is 13 kg -/
theorem zucchini_weight (carrots broccoli half_sold : ℝ) (h1 : carrots = 15) (h2 : broccoli = 8) (h3 : half_sold = 18) :
  ∃ zucchini : ℝ, (carrots + zucchini + broccoli) / 2 = half_sold ∧ zucchini = 13 := by
  sorry

end NUMINAMATH_CALUDE_zucchini_weight_l901_90193


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l901_90175

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {-1, 2, 3}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l901_90175


namespace NUMINAMATH_CALUDE_product_of_monomials_l901_90142

theorem product_of_monomials (x y : ℝ) : 2 * x * (-3 * x^2 * y^3) = -6 * x^3 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_monomials_l901_90142


namespace NUMINAMATH_CALUDE_new_student_weight_l901_90131

theorem new_student_weight (initial_count : ℕ) (replaced_weight : ℝ) (average_decrease : ℝ) :
  initial_count = 5 →
  replaced_weight = 72 →
  average_decrease = 12 →
  let new_weight := replaced_weight - initial_count * average_decrease
  new_weight = 12 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l901_90131


namespace NUMINAMATH_CALUDE_dana_marcus_pencil_difference_l901_90166

/-- Given that Dana has 15 more pencils than Jayden, Jayden has twice as many pencils as Marcus,
    and Jayden has 20 pencils, prove that Dana has 25 more pencils than Marcus. -/
theorem dana_marcus_pencil_difference :
  ∀ (dana jayden marcus : ℕ),
  dana = jayden + 15 →
  jayden = 2 * marcus →
  jayden = 20 →
  dana - marcus = 25 := by
sorry

end NUMINAMATH_CALUDE_dana_marcus_pencil_difference_l901_90166


namespace NUMINAMATH_CALUDE_chess_tournament_games_l901_90162

/-- Represents a chess tournament --/
structure ChessTournament where
  participants : ℕ
  total_games : ℕ
  games_per_player : ℕ
  h1 : total_games = participants * (participants - 1) / 2
  h2 : games_per_player = participants - 1

/-- Theorem: In a chess tournament with 20 participants and 190 total games, 
    each participant plays 19 games --/
theorem chess_tournament_games (t : ChessTournament) 
  (h_participants : t.participants = 20) 
  (h_total_games : t.total_games = 190) : 
  t.games_per_player = 19 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_l901_90162


namespace NUMINAMATH_CALUDE_a_eq_one_necessary_not_sufficient_l901_90148

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define what it means for two lines to be parallel
def parallel (a : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, l₁ a x y ↔ l₂ a (k * x) (k * y)

-- State the theorem
theorem a_eq_one_necessary_not_sufficient :
  (∀ a : ℝ, parallel a → a = 1) ∧ ¬(∀ a : ℝ, a = 1 → parallel a) :=
sorry

end NUMINAMATH_CALUDE_a_eq_one_necessary_not_sufficient_l901_90148


namespace NUMINAMATH_CALUDE_x_equals_y_when_t_is_half_l901_90118

theorem x_equals_y_when_t_is_half (t : ℚ) : 
  let x := 1 - 4 * t
  let y := 2 * t - 2
  x = y ↔ t = 1/2 := by
sorry

end NUMINAMATH_CALUDE_x_equals_y_when_t_is_half_l901_90118


namespace NUMINAMATH_CALUDE_matrix_power_4_l901_90176

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 1, 1]

theorem matrix_power_4 : A^4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l901_90176


namespace NUMINAMATH_CALUDE_prism_volume_l901_90137

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 20)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ (x y z : ℝ), x * y = side_area ∧ y * z = front_area ∧ x * z = bottom_area ∧ 
  x * y * z = 20 * Real.sqrt 4.8 :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_l901_90137


namespace NUMINAMATH_CALUDE_solve_for_y_l901_90112

theorem solve_for_y (x y : ℝ) (h1 : x = 100) (h2 : x^3*y - 3*x^2*y + 3*x*y = 3000000) : 
  y = 3000000 / 970299 := by sorry

end NUMINAMATH_CALUDE_solve_for_y_l901_90112


namespace NUMINAMATH_CALUDE_calculation_proof_l901_90122

theorem calculation_proof : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l901_90122


namespace NUMINAMATH_CALUDE_expected_monthly_profit_l901_90146

/-- Represents the color of a ball -/
inductive BallColor
| Yellow
| White

/-- Represents a ball with its color and label -/
structure Ball :=
  (color : BallColor)
  (label : Char)

/-- The set of all balls in the bag -/
def bag : Finset Ball := sorry

/-- The number of people drawing per day -/
def daily_draws : ℕ := 100

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The reward for drawing 3 balls of the same color -/
def same_color_reward : ℚ := 5

/-- The cost for drawing 3 balls of different colors -/
def diff_color_cost : ℚ := 1

/-- The probability of drawing 3 balls of the same color -/
def prob_same_color : ℚ := 2 / 20

/-- The probability of drawing 3 balls of different colors -/
def prob_diff_color : ℚ := 18 / 20

/-- The expected daily profit for the stall owner -/
def expected_daily_profit : ℚ :=
  daily_draws * (prob_diff_color * diff_color_cost - prob_same_color * same_color_reward)

/-- Theorem: The expected monthly profit for the stall owner is $1200 -/
theorem expected_monthly_profit :
  expected_daily_profit * days_in_month = 1200 := by sorry

end NUMINAMATH_CALUDE_expected_monthly_profit_l901_90146


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l901_90173

theorem roots_quadratic_equation (α β : ℝ) : 
  (∀ x : ℝ, x^2 + 4*x + 2 = 0 ↔ x = α ∨ x = β) → 
  α^3 + 14*β + 5 = -43 := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l901_90173


namespace NUMINAMATH_CALUDE_fraction_equality_problem_l901_90105

theorem fraction_equality_problem (y : ℝ) : 
  (4 + y) / (6 + y) = (2 + y) / (3 + y) ↔ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_problem_l901_90105


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_two_l901_90130

theorem trigonometric_expression_equals_two :
  (Real.cos (10 * π / 180) + Real.sqrt 3 * Real.sin (10 * π / 180)) /
  Real.sqrt (1 - Real.sin (50 * π / 180) ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_two_l901_90130


namespace NUMINAMATH_CALUDE_range_of_function_l901_90125

open Real

theorem range_of_function (f : ℝ → ℝ) (x : ℝ) :
  (f = fun x ↦ sin (x/2) * cos (x/2) + cos (x/2)^2) →
  (x ∈ Set.Ioo 0 (π/2)) →
  ∃ y, y ∈ Set.Ioc (1/2) ((Real.sqrt 2 + 1)/2) ∧ ∃ x, f x = y ∧
  ∀ z, (∃ x, f x = z) → z ∈ Set.Ioc (1/2) ((Real.sqrt 2 + 1)/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_function_l901_90125


namespace NUMINAMATH_CALUDE_total_participants_l901_90109

/-- Represents the exam scores and statistics -/
structure ExamStatistics where
  low_scorers : ℕ  -- Number of people scoring no more than 30
  low_avg : ℝ      -- Average score of low scorers
  high_scorers : ℕ -- Number of people scoring no less than 80
  high_avg : ℝ     -- Average score of high scorers
  above_30_avg : ℝ -- Average score of those scoring more than 30
  below_80_avg : ℝ -- Average score of those scoring less than 80

/-- Theorem stating the total number of participants in the exam -/
theorem total_participants (stats : ExamStatistics) 
  (h1 : stats.low_scorers = 153)
  (h2 : stats.low_avg = 24)
  (h3 : stats.high_scorers = 59)
  (h4 : stats.high_avg = 92)
  (h5 : stats.above_30_avg = 62)
  (h6 : stats.below_80_avg = 54) :
  stats.low_scorers + stats.high_scorers + 
  ((stats.low_scorers * (stats.below_80_avg - stats.low_avg) + 
    stats.high_scorers * (stats.high_avg - stats.above_30_avg)) / 
   (stats.above_30_avg - stats.below_80_avg)) = 1007 := by
  sorry


end NUMINAMATH_CALUDE_total_participants_l901_90109


namespace NUMINAMATH_CALUDE_min_distance_between_graphs_l901_90104

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (-2*x + 1)

/-- The logarithmic function -/
noncomputable def g (x : ℝ) : ℝ := (Real.log (-x - 1) - 3) / 2

/-- The symmetry line -/
noncomputable def l (x : ℝ) : ℝ := -x - 1

/-- Theorem stating the minimum distance between points on the two graphs -/
theorem min_distance_between_graphs :
  ∃ (P Q : ℝ × ℝ),
    (P.2 = f P.1) ∧ 
    (Q.2 = g Q.1) ∧
    (∀ (P' Q' : ℝ × ℝ), 
      P'.2 = f P'.1 → 
      Q'.2 = g Q'.1 → 
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = (Real.sqrt 2 * (4 + Real.log 2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_graphs_l901_90104


namespace NUMINAMATH_CALUDE_completing_square_sum_l901_90107

theorem completing_square_sum (a b : ℝ) : 
  (∀ x, x^2 + 6*x - 1 = 0 ↔ (x + a)^2 = b) → a + b = 13 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l901_90107


namespace NUMINAMATH_CALUDE_circles_intersection_parallel_lines_l901_90160

-- Define the basic geometric objects
variable (Circle1 Circle2 : Set (ℝ × ℝ))
variable (M K A B C D : ℝ × ℝ)

-- Define the conditions
axiom intersect_points : M ∈ Circle1 ∧ M ∈ Circle2 ∧ K ∈ Circle1 ∧ K ∈ Circle2
axiom line_AB : M.1 * B.2 - M.2 * B.1 = A.1 * B.2 - A.2 * B.1
axiom line_CD : K.1 * D.2 - K.2 * D.1 = C.1 * D.2 - C.2 * D.1
axiom A_in_Circle1 : A ∈ Circle1
axiom B_in_Circle2 : B ∈ Circle2
axiom C_in_Circle1 : C ∈ Circle1
axiom D_in_Circle2 : D ∈ Circle2

-- Define parallel lines
def parallel (p q r s : ℝ × ℝ) : Prop :=
  (p.1 - q.1) * (r.2 - s.2) = (p.2 - q.2) * (r.1 - s.1)

-- State the theorem
theorem circles_intersection_parallel_lines :
  parallel A C B D :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_parallel_lines_l901_90160


namespace NUMINAMATH_CALUDE_odd_totient_power_of_two_l901_90129

theorem odd_totient_power_of_two (n : ℕ) 
  (h_odd : Odd n)
  (h_phi_n : ∃ k : ℕ, Nat.totient n = 2^k)
  (h_phi_n_plus_one : ∃ m : ℕ, Nat.totient (n+1) = 2^m) :
  (∃ p : ℕ, n+1 = 2^p) ∨ n = 5 := by
sorry

end NUMINAMATH_CALUDE_odd_totient_power_of_two_l901_90129


namespace NUMINAMATH_CALUDE_lune_area_minus_triangle_l901_90117

/-- The area of a lune formed by two semicircles with diameters 3 and 4, 
    minus the area of an equilateral triangle inscribed in the smaller semicircle -/
theorem lune_area_minus_triangle (π : ℝ) : 
  let small_semicircle_area : ℝ := (1/2) * π * (3/2)^2
  let large_semicircle_area : ℝ := (1/2) * π * 2^2
  let lune_area : ℝ := small_semicircle_area - large_semicircle_area
  let triangle_side : ℝ := 3
  let triangle_height : ℝ := (3 * Real.sqrt 3) / 2
  let triangle_area : ℝ := (1/2) * triangle_side * triangle_height
  lune_area - triangle_area = -7/8 * π - 9 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_lune_area_minus_triangle_l901_90117


namespace NUMINAMATH_CALUDE_smallest_cyclic_divisible_by_1989_l901_90134

def is_cyclic_divisible_by_1989 (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10^n → ∀ i : ℕ, i < n → (k * 10^i + k / 10^(n - i)) % 1989 = 0

theorem smallest_cyclic_divisible_by_1989 :
  (∀ m < 48, ¬ is_cyclic_divisible_by_1989 m) ∧ is_cyclic_divisible_by_1989 48 :=
sorry

end NUMINAMATH_CALUDE_smallest_cyclic_divisible_by_1989_l901_90134


namespace NUMINAMATH_CALUDE_nicky_profit_l901_90165

def card_value_traded : ℕ := 8
def num_cards_traded : ℕ := 2
def card_value_received : ℕ := 21

def profit : ℕ := card_value_received - (card_value_traded * num_cards_traded)

theorem nicky_profit :
  profit = 5 := by sorry

end NUMINAMATH_CALUDE_nicky_profit_l901_90165


namespace NUMINAMATH_CALUDE_min_c_over_d_l901_90143

theorem min_c_over_d (x C D : ℝ) (hx : x ≠ 0) (hC : C > 0) (hD : D > 0)
  (hxC : x^4 + 1/x^4 = C) (hxD : x^2 - 1/x^2 = D) :
  ∃ (m : ℝ), (∀ x' C' D', x' ≠ 0 → C' > 0 → D' > 0 → 
    x'^4 + 1/x'^4 = C' → x'^2 - 1/x'^2 = D' → C' / D' ≥ m) ∧ 
  (∃ x₀ C₀ D₀, x₀ ≠ 0 ∧ C₀ > 0 ∧ D₀ > 0 ∧ 
    x₀^4 + 1/x₀^4 = C₀ ∧ x₀^2 - 1/x₀^2 = D₀ ∧ C₀ / D₀ = m) ∧
  m = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_c_over_d_l901_90143


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_larger_base_l901_90178

/-- An isosceles trapezoid with given measurements -/
structure IsoscelesTrapezoid where
  leg : ℝ
  smallerBase : ℝ
  diagonal : ℝ
  largerBase : ℝ

/-- The isosceles trapezoid satisfies the given conditions -/
def satisfiesConditions (t : IsoscelesTrapezoid) : Prop :=
  t.leg = 10 ∧ t.smallerBase = 6 ∧ t.diagonal = 14

/-- Theorem: The larger base of the isosceles trapezoid is 16 -/
theorem isosceles_trapezoid_larger_base
  (t : IsoscelesTrapezoid)
  (h : satisfiesConditions t) :
  t.largerBase = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_larger_base_l901_90178


namespace NUMINAMATH_CALUDE_min_c_value_l901_90128

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! p : ℝ × ℝ, p.1 * 2 + p.2 = 2021 ∧ p.2 = |p.1 - a| + |p.1 - b| + |p.1 - c|) :
  c ≥ 1011 :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l901_90128


namespace NUMINAMATH_CALUDE_elongation_rate_improved_l901_90108

def elongation_rate_comparison (x y : Fin 10 → ℝ) : Prop :=
  let z : Fin 10 → ℝ := fun i => x i - y i
  let z_mean : ℝ := (Finset.sum Finset.univ (fun i => z i)) / 10
  let z_variance : ℝ := (Finset.sum Finset.univ (fun i => (z i - z_mean)^2)) / 10
  z_mean = 11 ∧ 
  z_variance = 61 ∧ 
  z_mean ≥ 2 * Real.sqrt (z_variance / 10)

theorem elongation_rate_improved (x y : Fin 10 → ℝ) 
  (h : elongation_rate_comparison x y) : 
  ∃ (z_mean z_variance : ℝ), 
    z_mean = 11 ∧ 
    z_variance = 61 ∧ 
    z_mean ≥ 2 * Real.sqrt (z_variance / 10) :=
by
  sorry

end NUMINAMATH_CALUDE_elongation_rate_improved_l901_90108


namespace NUMINAMATH_CALUDE_root_condition_implies_k_range_l901_90182

theorem root_condition_implies_k_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
    2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
    |x₁ - 2*n| = k * Real.sqrt x₁ ∧
    |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_k_range_l901_90182


namespace NUMINAMATH_CALUDE_fraction_undefined_l901_90121

theorem fraction_undefined (x : ℚ) : (2 * x + 1 = 0) ↔ (x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_l901_90121


namespace NUMINAMATH_CALUDE_account_balance_after_transactions_l901_90170

/-- Calculates the final account balance after a series of transactions --/
def finalBalance (initialBalance : ℚ) 
  (transfer1 transfer2 transfer3 transfer4 transfer5 : ℚ)
  (serviceCharge1 serviceCharge2 serviceCharge3 serviceCharge4 serviceCharge5 : ℚ) : ℚ :=
  initialBalance - transfer1 - (transfer3 + serviceCharge3) - (transfer5 + serviceCharge5)

/-- Theorem stating the final account balance after the given transactions --/
theorem account_balance_after_transactions 
  (initialBalance : ℚ)
  (transfer1 transfer2 transfer3 transfer4 transfer5 : ℚ)
  (serviceCharge1 serviceCharge2 serviceCharge3 serviceCharge4 serviceCharge5 : ℚ)
  (h1 : initialBalance = 400)
  (h2 : transfer1 = 90)
  (h3 : transfer2 = 60)
  (h4 : transfer3 = 50)
  (h5 : transfer4 = 120)
  (h6 : transfer5 = 200)
  (h7 : serviceCharge1 = 0.02 * transfer1)
  (h8 : serviceCharge2 = 0.02 * transfer2)
  (h9 : serviceCharge3 = 0.02 * transfer3)
  (h10 : serviceCharge4 = 0.025 * transfer4)
  (h11 : serviceCharge5 = 0.03 * transfer5) :
  finalBalance initialBalance transfer1 transfer2 transfer3 transfer4 transfer5
    serviceCharge1 serviceCharge2 serviceCharge3 serviceCharge4 serviceCharge5 = 53 := by
  sorry


end NUMINAMATH_CALUDE_account_balance_after_transactions_l901_90170


namespace NUMINAMATH_CALUDE_total_cost_is_1046_l901_90138

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 349 / 100

/-- The number of sandwiches -/
def num_sandwiches : ℕ := 2

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 87 / 100

/-- The number of sodas -/
def num_sodas : ℕ := 4

/-- The total cost of the order -/
def total_cost : ℚ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem total_cost_is_1046 : total_cost = 1046 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_1046_l901_90138


namespace NUMINAMATH_CALUDE_ratio_problem_l901_90102

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (x + y) = 4 / 5) : 
  x / y = 14 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l901_90102


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l901_90123

/-- Sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℕ) (aₙ : ℕ) : ℕ := 
  let n := aₙ - a₁ + 1
  n * (a₁ + aₙ) / 2

theorem sequence_sum_problem : 
  (arithmeticSum 2001 2093) - (arithmeticSum 221 313) + (arithmeticSum 401 493) = 207141 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l901_90123


namespace NUMINAMATH_CALUDE_polynomial_characterization_l901_90135

def has_same_prime_divisors (a b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ a ↔ p ∣ b)

def is_valid_polynomial (f : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → has_same_prime_divisors n (Int.natAbs (f n))

theorem polynomial_characterization :
  ∀ f : ℕ → ℤ,
  (∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, f n = n^k ∨ f n = -(n^k))) ↔
  is_valid_polynomial f :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l901_90135


namespace NUMINAMATH_CALUDE_inverse_f_at_3_l901_90194

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (y : ℝ), y < 0 ∧ f y = 3 ∧ ∀ (z : ℝ), z < 0 ∧ f z = 3 → z = y :=
by sorry

end NUMINAMATH_CALUDE_inverse_f_at_3_l901_90194


namespace NUMINAMATH_CALUDE_average_salary_feb_to_may_l901_90187

theorem average_salary_feb_to_may (
  avg_jan_to_apr : ℝ) 
  (salary_may : ℝ)
  (salary_jan : ℝ)
  (h1 : avg_jan_to_apr = 8000)
  (h2 : salary_may = 6500)
  (h3 : salary_jan = 4700) :
  (4 * avg_jan_to_apr - salary_jan + salary_may) / 4 = 8450 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_feb_to_may_l901_90187


namespace NUMINAMATH_CALUDE_birdhouse_cost_theorem_l901_90191

/-- Calculates the total cost of building birdhouses -/
def total_cost_birdhouses (small_count large_count : ℕ) 
  (small_plank_req large_plank_req : ℕ) 
  (small_nail_req large_nail_req : ℕ) 
  (small_plank_cost large_plank_cost nail_cost : ℚ) 
  (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let total_small_planks := small_count * small_plank_req
  let total_large_planks := large_count * large_plank_req
  let total_nails := small_count * small_nail_req + large_count * large_nail_req
  let plank_cost := total_small_planks * small_plank_cost + total_large_planks * large_plank_cost
  let nail_cost_before_discount := total_nails * nail_cost
  let nail_cost_after_discount := 
    if total_nails > discount_threshold
    then nail_cost_before_discount * (1 - discount_rate)
    else nail_cost_before_discount
  plank_cost + nail_cost_after_discount

theorem birdhouse_cost_theorem :
  total_cost_birdhouses 3 2 7 10 20 36 3 5 (5/100) 100 (1/10) = 16894/100 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_cost_theorem_l901_90191


namespace NUMINAMATH_CALUDE_quadratic_intersection_intersection_points_l901_90198

/-- Quadratic function f(x) = x^2 - 6x + 2m - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + 2*m - 1

theorem quadratic_intersection (m : ℝ) :
  (∀ x, f m x ≠ 0) ↔ m > 5 :=
sorry

theorem intersection_points :
  let m : ℝ := -3
  (∃ x, f m x = 0 ∧ (x = -1 ∨ x = 7)) ∧
  (f m 0 = -7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_intersection_points_l901_90198


namespace NUMINAMATH_CALUDE_sum_five_consecutive_odds_mod_12_l901_90101

theorem sum_five_consecutive_odds_mod_12 (n : ℕ) : 
  (((2*n + 1) + (2*n + 3) + (2*n + 5) + (2*n + 7) + (2*n + 9)) % 12) = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_odds_mod_12_l901_90101


namespace NUMINAMATH_CALUDE_intersection_and_angle_condition_l901_90103

-- Define the lines
def l1 (x y : ℝ) : Prop := x + y + 1 = 0
def l2 (x y : ℝ) : Prop := 5 * x - y - 1 = 0
def l3 (x y : ℝ) : Prop := 3 * x + 2 * y + 1 = 0

-- Define the result lines
def result1 (x y : ℝ) : Prop := x + 5 * y + 5 = 0
def result2 (x y : ℝ) : Prop := 5 * x - y - 1 = 0

-- Define the 45° angle condition
def angle_45_deg (m1 m2 : ℝ) : Prop := (m1 - m2) / (1 + m1 * m2) = 1 || (m1 - m2) / (1 + m1 * m2) = -1

-- Main theorem
theorem intersection_and_angle_condition :
  ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧
  (∃ (m : ℝ), (angle_45_deg m (-3/2)) ∧
    ((result1 x y ∧ m = -1/5) ∨ (result2 x y ∧ m = 5))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_angle_condition_l901_90103


namespace NUMINAMATH_CALUDE_polynomial_expansion_property_l901_90150

theorem polynomial_expansion_property (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 + x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₂ - a₁ + a₄ - a₃ = -15 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_property_l901_90150
