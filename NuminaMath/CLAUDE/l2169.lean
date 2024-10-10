import Mathlib

namespace tan_two_alpha_plus_pi_fourth_l2169_216964

theorem tan_two_alpha_plus_pi_fourth (α : ℝ) 
  (h : (2 * (Real.cos α)^2 + Real.cos (π/2 + 2*α) - 1) / (Real.sqrt 2 * Real.sin (2*α + π/4)) = 4) : 
  Real.tan (2*α + π/4) = 1/4 := by
  sorry

end tan_two_alpha_plus_pi_fourth_l2169_216964


namespace square_of_1024_l2169_216923

theorem square_of_1024 : (1024 : ℕ)^2 = 1048576 := by
  sorry

end square_of_1024_l2169_216923


namespace trajectory_of_C_l2169_216965

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ :=
  dist t.A t.B + dist t.B t.C + dist t.C t.A

-- Theorem statement
theorem trajectory_of_C (x y : ℝ) :
  let t := Triangle.mk (0, 2) (0, -2) (x, y)
  perimeter t = 10 ∧ x ≠ 0 →
  x^2 / 5 + y^2 / 9 = 1 :=
by sorry

end trajectory_of_C_l2169_216965


namespace min_value_theorem_l2169_216936

theorem min_value_theorem (n : ℕ+) (a : ℝ) (x : ℝ) (ha : a > 0) (hx : x > 0) :
  (a^n.val + x^n.val) * (a + x)^n.val / x^n.val ≥ 2^(n.val + 1) * a^n.val :=
by sorry

end min_value_theorem_l2169_216936


namespace four_liars_in_group_l2169_216997

/-- Represents a person who is either a knight or a liar -/
inductive Person
  | Knight
  | Liar

/-- Represents an answer to the question "How many liars are among you?" -/
def Answer := Fin 5

/-- A function that determines whether a person is telling the truth given their answer and the actual number of liars -/
def isTellingTruth (p : Person) (answer : Answer) (actualLiars : Nat) : Prop :=
  match p with
  | Person.Knight => answer.val + 1 = actualLiars
  | Person.Liar => answer.val + 1 ≠ actualLiars

/-- The main theorem -/
theorem four_liars_in_group (group : Fin 5 → Person) (answers : Fin 5 → Answer) 
    (h_distinct : ∀ i j, i ≠ j → answers i ≠ answers j) :
    (∃ (actualLiars : Nat), actualLiars = 4 ∧ 
      ∀ i, isTellingTruth (group i) (answers i) actualLiars) := by
  sorry

end four_liars_in_group_l2169_216997


namespace right_triangle_sets_l2169_216994

theorem right_triangle_sets :
  let set1 : Fin 3 → ℝ := ![3, 4, 5]
  let set2 : Fin 3 → ℝ := ![9, 12, 15]
  let set3 : Fin 3 → ℝ := ![Real.sqrt 3, 2, Real.sqrt 5]
  let set4 : Fin 3 → ℝ := ![0.3, 0.4, 0.5]

  (set1 0)^2 + (set1 1)^2 = (set1 2)^2 ∧
  (set2 0)^2 + (set2 1)^2 = (set2 2)^2 ∧
  (set3 0)^2 + (set3 1)^2 ≠ (set3 2)^2 ∧
  (set4 0)^2 + (set4 1)^2 = (set4 2)^2 :=
by sorry

end right_triangle_sets_l2169_216994


namespace tuning_day_method_pi_approximation_l2169_216992

/-- The Tuning Day Method function -/
def tuningDayMethod (a b c d : ℕ) : ℚ := (b + d) / (a + c)

/-- Check if a fraction is simpler than another -/
def isSimpler (a b c d : ℕ) : Bool :=
  a + b < c + d ∨ (a + b = c + d ∧ a < c)

theorem tuning_day_method_pi_approximation :
  let initial_lower : ℚ := 31 / 10
  let initial_upper : ℚ := 49 / 15
  let step1 : ℚ := tuningDayMethod 10 31 15 49
  let step2 : ℚ := tuningDayMethod 10 31 5 16
  let step3 : ℚ := tuningDayMethod 15 47 5 16
  let step4 : ℚ := tuningDayMethod 15 47 20 63
  initial_lower < Real.pi ∧ Real.pi < initial_upper ∧
  step1 = 16 / 5 ∧
  step2 = 47 / 15 ∧
  step3 = 63 / 20 ∧
  step4 = 22 / 7 ∧
  isSimpler 22 7 63 20 ∧
  isSimpler 22 7 47 15 ∧
  isSimpler 22 7 16 5 ∧
  47 / 15 < Real.pi ∧ Real.pi < 22 / 7 :=
by sorry

end tuning_day_method_pi_approximation_l2169_216992


namespace sum_of_fourth_and_fifth_terms_l2169_216902

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  (a 0 = 2048) ∧ 
  (a 1 = 512) ∧ 
  (a 2 = 128) ∧ 
  (a 5 = 2) ∧ 
  ∀ n, a (n + 1) = a n * (a 1 / a 0)

/-- The sum of the fourth and fifth terms in the sequence is 40 -/
theorem sum_of_fourth_and_fifth_terms (a : ℕ → ℚ) 
  (h : geometric_sequence a) : a 3 + a 4 = 40 := by
  sorry

end sum_of_fourth_and_fifth_terms_l2169_216902


namespace triangle_angle_measure_l2169_216972

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  a = 2 * Real.sqrt 3 →
  c = 2 * Real.sqrt 2 →
  A = π / 3 →
  (a / Real.sin A = c / Real.sin C) →
  C < π / 2 →
  C = π / 4 :=
by sorry

end triangle_angle_measure_l2169_216972


namespace tangent_circles_t_value_l2169_216971

-- Define the circles
def circle1 (t : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = t^2
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y + 24 = 0

-- Define tangency
def tangent (t : ℝ) : Prop := ∃ x y : ℝ, circle1 t x y ∧ circle2 x y

-- Theorem statement
theorem tangent_circles_t_value :
  ∀ t : ℝ, t > 0 → tangent t → t = 4 :=
sorry

end tangent_circles_t_value_l2169_216971


namespace complement_of_M_in_S_l2169_216953

def S : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_S :
  S \ M = {3, 5} := by sorry

end complement_of_M_in_S_l2169_216953


namespace negation_of_all_nonnegative_squares_l2169_216942

theorem negation_of_all_nonnegative_squares (p : Prop) : 
  (p ↔ ∀ x : ℝ, x^2 ≥ 0) → (¬p ↔ ∃ x : ℝ, x^2 < 0) := by sorry

end negation_of_all_nonnegative_squares_l2169_216942


namespace triangular_number_gcd_bound_triangular_number_gcd_achieves_three_l2169_216948

def triangular_number (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

theorem triangular_number_gcd_bound (n : ℕ+) : 
  Nat.gcd (6 * triangular_number n) (n + 1) ≤ 3 :=
sorry

theorem triangular_number_gcd_achieves_three : 
  ∃ n : ℕ+, Nat.gcd (6 * triangular_number n) (n + 1) = 3 :=
sorry

end triangular_number_gcd_bound_triangular_number_gcd_achieves_three_l2169_216948


namespace mcgees_bakery_pies_l2169_216969

theorem mcgees_bakery_pies (smiths_pies mcgees_pies : ℕ) : 
  smiths_pies = 70 → 
  smiths_pies = 4 * mcgees_pies + 6 → 
  mcgees_pies = 16 := by
sorry

end mcgees_bakery_pies_l2169_216969


namespace square_measurement_error_l2169_216919

theorem square_measurement_error (actual_side : ℝ) (measured_side : ℝ) 
  (h : measured_side ^ 2 = 1.0816 * actual_side ^ 2) : 
  (measured_side - actual_side) / actual_side = 0.04 := by
  sorry

end square_measurement_error_l2169_216919


namespace triangle_angle_expression_minimum_l2169_216920

theorem triangle_angle_expression_minimum (A B C : Real) 
  (h_triangle : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) : 
  (1 / (Real.sin A)^2) + (1 / (Real.sin B)^2) + (4 / (1 + Real.sin C)) ≥ 16 - 8 * Real.sqrt 2 := by
  sorry

end triangle_angle_expression_minimum_l2169_216920


namespace negation_of_existence_proposition_l2169_216970

theorem negation_of_existence_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end negation_of_existence_proposition_l2169_216970


namespace jimin_tape_length_l2169_216977

-- Define the conversion factor from cm to mm
def cm_to_mm : ℝ := 10

-- Define Jungkook's tape length in cm
def jungkook_tape_cm : ℝ := 45

-- Define the difference between Jimin's and Jungkook's tape lengths in mm
def tape_difference_mm : ℝ := 26

-- State the theorem
theorem jimin_tape_length :
  (jungkook_tape_cm * cm_to_mm + tape_difference_mm) / cm_to_mm = 47.6 := by
  sorry

end jimin_tape_length_l2169_216977


namespace valid_numbers_count_l2169_216949

/-- Converts a base-10 number to base-12 --/
def toBase12 (n : ℕ) : ℕ := sorry

/-- Checks if a base-12 number uses only digits 0-9 --/
def usesOnlyDigits0to9 (n : ℕ) : Bool := sorry

/-- Counts numbers up to n (base-10) that use only digits 0-9 in base-12 --/
def countValidNumbers (n : ℕ) : ℕ := sorry

theorem valid_numbers_count :
  countValidNumbers 1200 = 90 := by sorry

end valid_numbers_count_l2169_216949


namespace bike_shop_profit_is_8206_l2169_216966

/-- Represents the profit calculation for Jim's bike shop -/
def bike_shop_profit (tire_repair_price tire_repair_cost tire_repairs_count
                      chain_repair_price chain_repair_cost chain_repairs_count
                      overhaul_price overhaul_cost overhaul_count
                      retail_sales retail_cost
                      parts_discount_threshold parts_discount_rate
                      tax_rate fixed_expenses : ℚ) : ℚ :=
  let total_income := tire_repair_price * tire_repairs_count +
                      chain_repair_price * chain_repairs_count +
                      overhaul_price * overhaul_count +
                      retail_sales

  let total_parts_cost := tire_repair_cost * tire_repairs_count +
                          chain_repair_cost * chain_repairs_count +
                          overhaul_cost * overhaul_count

  let parts_discount := if total_parts_cost ≥ parts_discount_threshold
                        then total_parts_cost * parts_discount_rate
                        else 0

  let final_parts_cost := total_parts_cost - parts_discount

  let profit_before_tax := total_income - final_parts_cost - retail_cost

  let taxes := total_income * tax_rate

  profit_before_tax - taxes - fixed_expenses

/-- Theorem stating that the bike shop's profit is $8206 given the specified conditions -/
theorem bike_shop_profit_is_8206 :
  bike_shop_profit 20 5 300
                   75 25 50
                   300 50 8
                   2000 1200
                   2500 0.1
                   0.06 4000 = 8206 := by sorry

end bike_shop_profit_is_8206_l2169_216966


namespace absolute_value_inequality_l2169_216921

theorem absolute_value_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
by sorry

end absolute_value_inequality_l2169_216921


namespace morning_run_distance_l2169_216960

/-- Represents a person's daily activities and distances --/
structure DailyActivities where
  n : ℕ  -- number of stores visited
  x : ℝ  -- morning run distance
  total_distance : ℝ  -- total distance for the day
  bike_distance : ℝ  -- evening bike ride distance

/-- Theorem stating the relationship between morning run distance and other factors --/
theorem morning_run_distance (d : DailyActivities) 
  (h1 : d.total_distance = 18) 
  (h2 : d.bike_distance = 12) 
  (h3 : d.total_distance = d.x + 2 * d.n * d.x + d.bike_distance) :
  d.x = 6 / (1 + 2 * d.n) := by
  sorry

end morning_run_distance_l2169_216960


namespace monotonic_quadratic_l2169_216944

/-- A function f is monotonic on an interval [a,b] if it is either 
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The statement of the problem -/
theorem monotonic_quadratic (a : ℝ) :
  IsMonotonic (fun x => x^2 + (1-a)*x + 3) 1 4 ↔ a ≥ 9 ∨ a ≤ 3 :=
sorry

end monotonic_quadratic_l2169_216944


namespace happy_point_range_l2169_216945

theorem happy_point_range (a : ℝ) :
  (∃ x ∈ Set.Icc (-3 : ℝ) (-3/2), a * x^2 - 2*x - 2*a - 3/2 = -x) →
  a ∈ Set.Icc (-1/4 : ℝ) 0 := by
sorry

end happy_point_range_l2169_216945


namespace sphere_surface_area_of_prism_inscribed_l2169_216946

/-- Given a rectangular prism with adjacent face areas of 2, 3, and 6,
    and all vertices lying on the same spherical surface,
    prove that the surface area of this sphere is 14π. -/
theorem sphere_surface_area_of_prism_inscribed (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b = 6 → b * c = 2 → a * c = 3 →
  (4 : ℝ) * Real.pi * ((a^2 + b^2 + c^2) / 4) = 14 * Real.pi := by
  sorry

#check sphere_surface_area_of_prism_inscribed

end sphere_surface_area_of_prism_inscribed_l2169_216946


namespace prism_18_edges_8_faces_l2169_216976

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  (p.edges / 3) + 2

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ p : Prism, p.edges = 18 → num_faces p = 8 := by
  sorry

end prism_18_edges_8_faces_l2169_216976


namespace janet_muffins_count_l2169_216980

theorem janet_muffins_count :
  ∀ (muffin_cost : ℚ) (paid : ℚ) (change : ℚ),
    muffin_cost = 75 / 100 →
    paid = 20 →
    change = 11 →
    (paid - change) / muffin_cost = 12 :=
by sorry

end janet_muffins_count_l2169_216980


namespace expression_evaluation_l2169_216905

theorem expression_evaluation :
  let a : ℤ := -1
  (a^2 + 1) - 3*a*(a - 1) + 2*(a^2 + a - 1) = -6 :=
by sorry

end expression_evaluation_l2169_216905


namespace fraction_inequality_l2169_216962

theorem fraction_inequality (a b c d : ℕ+) (h1 : a + c < 1988) 
  (h2 : (1 : ℚ) - a / b - c / d > 0) : (1 : ℚ) - a / b - c / d > 1 / (1988^3) := by
  sorry

end fraction_inequality_l2169_216962


namespace greatest_possible_award_l2169_216959

theorem greatest_possible_award (total_prize : ℝ) (num_winners : ℕ) (min_award : ℝ) 
  (prize_fraction : ℝ) (winner_fraction : ℝ) :
  total_prize = 400 →
  num_winners = 20 →
  min_award = 20 →
  prize_fraction = 2/5 →
  winner_fraction = 3/5 →
  ∃ (max_award : ℝ), 
    max_award = 100 ∧ 
    max_award ≤ total_prize ∧
    max_award ≥ min_award ∧
    (∀ (award : ℝ), 
      award ≤ total_prize ∧ 
      award ≥ min_award → 
      award ≤ max_award) ∧
    (prize_fraction * total_prize ≤ winner_fraction * num_winners * min_award) :=
by
  sorry

end greatest_possible_award_l2169_216959


namespace adam_remaining_candy_l2169_216911

/-- Calculates the number of candy pieces Adam has left after giving some boxes away. -/
def remaining_candy_pieces (initial_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  (initial_boxes - given_away_boxes) * pieces_per_box

/-- Proves that Adam has 36 pieces of candy left. -/
theorem adam_remaining_candy :
  remaining_candy_pieces 13 7 6 = 36 := by
  sorry

#eval remaining_candy_pieces 13 7 6

end adam_remaining_candy_l2169_216911


namespace initial_rate_is_36_l2169_216952

/-- Represents the production of cogs on an assembly line with two phases -/
def cog_production (initial_rate : ℝ) : Prop :=
  let initial_order := 60
  let second_order := 60
  let increased_rate := 60
  let total_cogs := initial_order + second_order
  let initial_time := initial_order / initial_rate
  let second_time := second_order / increased_rate
  let total_time := initial_time + second_time
  let average_output := 45
  (total_cogs / total_time) = average_output

/-- The theorem stating that the initial production rate is 36 cogs per hour -/
theorem initial_rate_is_36 : 
  ∃ (rate : ℝ), cog_production rate ∧ rate = 36 :=
sorry

end initial_rate_is_36_l2169_216952


namespace challenge_result_l2169_216995

theorem challenge_result (x : ℕ) : 3 * (3 * (x + 1) + 3) = 63 := by
  sorry

#check challenge_result

end challenge_result_l2169_216995


namespace tim_picked_five_pears_l2169_216910

/-- The number of pears Sara picked -/
def sara_pears : ℕ := 6

/-- The total number of pears picked by Sara and Tim -/
def total_pears : ℕ := 11

/-- The number of pears Tim picked -/
def tim_pears : ℕ := total_pears - sara_pears

theorem tim_picked_five_pears : tim_pears = 5 := by
  sorry

end tim_picked_five_pears_l2169_216910


namespace probability_of_specific_match_l2169_216929

/-- Calculates the probability of two specific players facing each other in a tournament. -/
theorem probability_of_specific_match (n : ℕ) (h : n = 26) : 
  (n - 1 : ℚ) / (n * (n - 1) / 2) = 1 / 13 := by
  sorry

#check probability_of_specific_match

end probability_of_specific_match_l2169_216929


namespace train_platform_passage_time_train_platform_passage_time_specific_l2169_216916

/-- Calculates the time taken for a train to pass a platform given its speed, 
    the platform length, and the time taken to pass a stationary man. -/
theorem train_platform_passage_time 
  (train_speed_kmh : ℝ) 
  (platform_length : ℝ) 
  (time_pass_man : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let train_length := train_speed_ms * time_pass_man
  let total_distance := platform_length + train_length
  let time_pass_platform := total_distance / train_speed_ms
  time_pass_platform

/-- Proves that given the specific conditions, the time taken to pass 
    the platform is approximately 30 seconds. -/
theorem train_platform_passage_time_specific : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_platform_passage_time 54 150.012 20 - 30| < ε :=
sorry

end train_platform_passage_time_train_platform_passage_time_specific_l2169_216916


namespace students_exceed_rabbits_l2169_216938

theorem students_exceed_rabbits :
  let classrooms : ℕ := 5
  let students_per_classroom : ℕ := 23
  let rabbits_per_classroom : ℕ := 3
  let total_students : ℕ := classrooms * students_per_classroom
  let total_rabbits : ℕ := classrooms * rabbits_per_classroom
  total_students - total_rabbits = 100 := by
sorry

end students_exceed_rabbits_l2169_216938


namespace new_person_weight_example_l2169_216931

/-- Calculates the weight of a new person given the initial number of persons,
    the average weight increase, and the weight of the replaced person. -/
def new_person_weight (initial_count : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * avg_increase

theorem new_person_weight_example :
  new_person_weight 7 6.2 76 = 119.4 := by
  sorry

end new_person_weight_example_l2169_216931


namespace emily_square_subtraction_l2169_216935

theorem emily_square_subtraction : 49^2 = 50^2 - 99 := by
  sorry

end emily_square_subtraction_l2169_216935


namespace bushel_weight_is_56_l2169_216917

/-- The weight of a bushel of corn in pounds -/
def bushel_weight : ℝ := 56

/-- The weight of an individual ear of corn in pounds -/
def ear_weight : ℝ := 0.5

/-- The number of bushels Clyde picked -/
def bushels_picked : ℕ := 2

/-- The number of individual corn cobs Clyde picked -/
def cobs_picked : ℕ := 224

/-- Theorem: The weight of a bushel of corn is 56 pounds -/
theorem bushel_weight_is_56 : 
  bushel_weight = (ear_weight * cobs_picked) / bushels_picked :=
sorry

end bushel_weight_is_56_l2169_216917


namespace max_value_of_product_l2169_216937

theorem max_value_of_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  x^2 * y^3 * z ≤ 1 / 3888 :=
sorry

end max_value_of_product_l2169_216937


namespace hyperbola_area_ratio_l2169_216941

noncomputable def hyperbola_ratio (a b : ℝ) (F₁ F₂ A B : ℝ × ℝ) : Prop :=
  let x := λ p : ℝ × ℝ => p.1
  let y := λ p : ℝ × ℝ => p.2
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((x p - x q)^2 + (y p - y q)^2)
  let area := λ p q r : ℝ × ℝ => abs ((x q - x p) * (y r - y p) - (x r - x p) * (y q - y p)) / 2
  (∀ p : ℝ × ℝ, (x p)^2 / a^2 - (y p)^2 / b^2 = 1 → 
    (x p - x F₁) * (x p - x F₂) + (y p - y F₁) * (y p - y F₂) = a^2 - b^2) ∧
  (x F₁ = -Real.sqrt (a^2 + b^2) ∧ y F₁ = 0) ∧
  (x F₂ = Real.sqrt (a^2 + b^2) ∧ y F₂ = 0) ∧
  ((x A)^2 / a^2 - (y A)^2 / b^2 = 1) ∧
  ((x B)^2 / a^2 - (y B)^2 / b^2 = 1) ∧
  (y B - y A) * (x A - x F₁) = (x B - x A) * (y A - y F₁) ∧
  dist A F₁ / dist A F₂ = 1/2 →
  area A F₁ F₂ / area A B F₂ = 4/9

theorem hyperbola_area_ratio : 
  hyperbola_ratio 3 4 (-5, 0) (5, 0) (-27/5, 8*Real.sqrt 14/5) (0, 0) :=
sorry

end hyperbola_area_ratio_l2169_216941


namespace trees_in_yard_l2169_216981

/-- The number of trees in a yard with given conditions -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem stating the number of trees in the yard under given conditions -/
theorem trees_in_yard :
  let yard_length : ℕ := 150
  let tree_distance : ℕ := 15
  number_of_trees yard_length tree_distance = 11 := by
  sorry

end trees_in_yard_l2169_216981


namespace max_sum_is_1120_l2169_216928

/-- Represents a splitting operation on a pile of coins -/
structure Split :=
  (a : ℕ) (b : ℕ) (c : ℕ)
  (h1 : a > 1)
  (h2 : b ≥ 1)
  (h3 : c ≥ 1)
  (h4 : a = b + c)

/-- Represents the state of the coin piles -/
structure PileState :=
  (piles : List ℕ)
  (board_sum : ℕ)

/-- Performs a single split operation on a pile state -/
def split_pile (state : PileState) (split : Split) : PileState :=
  sorry

/-- Checks if the splitting process is complete -/
def is_complete (state : PileState) : Bool :=
  state.piles.length == 15 && state.piles.all (· == 1)

/-- Finds the maximum possible board sum after splitting 15 coins into 15 piles -/
def max_board_sum : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem max_sum_is_1120 :
  max_board_sum = 1120 :=
sorry

end max_sum_is_1120_l2169_216928


namespace almeriense_polynomial_characterization_l2169_216906

/-- A polynomial is almeriense if it has the form x³ + ax² + bx + a
    and its three roots are positive real numbers in arithmetic progression. -/
def IsAlmeriense (p : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ),
    (∀ x, p x = x^3 + a*x^2 + b*x + a) ∧
    (∃ (r₁ r₂ r₃ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
      r₂ - r₁ = r₃ - r₂ ∧
      (∀ x, p x = (x - r₁) * (x - r₂) * (x - r₃)))

theorem almeriense_polynomial_characterization :
  ∀ p : ℝ → ℝ,
    IsAlmeriense p →
    p (7/4) = 0 →
    ((∀ x, p x = x^3 - (21/4)*x^2 + (73/8)*x - 21/4) ∨
     (∀ x, p x = x^3 - (291/56)*x^2 + (14113/1568)*x - 291/56)) :=
by sorry

end almeriense_polynomial_characterization_l2169_216906


namespace cubic_three_zeros_l2169_216961

/-- The cubic function f(x) = x^3 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- Theorem: The cubic function f(x) = x^3 + ax + 2 has exactly 3 real zeros if and only if a < -3 -/
theorem cubic_three_zeros (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ a < -3 :=
sorry

end cubic_three_zeros_l2169_216961


namespace find_a_l2169_216909

/-- The system of equations -/
def system (a b m : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y = 2 ∧ m * x - 7 * y = -8

/-- Xiao Li's solution -/
def solution_li (a b : ℝ) : Prop :=
  a * (-2) + b * 3 = 2

/-- Xiao Zhang's solution -/
def solution_zhang (a b : ℝ) : Prop :=
  a * (-2) + b * 2 = 2

/-- Theorem stating that if both solutions satisfy the first equation, then a = -1 -/
theorem find_a (a b m : ℝ) : solution_li a b ∧ solution_zhang a b → a = -1 := by
  sorry

end find_a_l2169_216909


namespace interest_difference_approx_l2169_216979

def principal : ℝ := 147.69
def rate : ℝ := 0.15
def time1 : ℝ := 3.5
def time2 : ℝ := 10

def interest (p r t : ℝ) : ℝ := p * r * t

theorem interest_difference_approx :
  ∃ ε > 0, ε < 0.001 ∧ 
  |interest principal rate time2 - interest principal rate time1 - 143.998| < ε :=
sorry

end interest_difference_approx_l2169_216979


namespace greatest_three_digit_multiple_of_17_l2169_216907

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l2169_216907


namespace passengers_left_is_200_l2169_216924

/-- The number of minutes between train arrivals -/
def train_interval : ℕ := 5

/-- The number of passengers each train takes -/
def passengers_taken : ℕ := 320

/-- The total number of different passengers stepping on and off trains in one hour -/
def total_passengers : ℕ := 6240

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of passengers each train leaves at the station -/
def passengers_left : ℕ := (total_passengers - (minutes_per_hour / train_interval * passengers_taken)) / (minutes_per_hour / train_interval)

theorem passengers_left_is_200 : passengers_left = 200 := by
  sorry

end passengers_left_is_200_l2169_216924


namespace largest_four_digit_number_with_conditions_l2169_216989

theorem largest_four_digit_number_with_conditions : 
  ∀ n : ℕ, 
  n ≤ 9999 ∧ n ≥ 1000 ∧ 
  ∃ k : ℕ, n = 11 * k + 2 ∧
  ∃ m : ℕ, n = 7 * m + 4 
  → n ≤ 9979 :=
by sorry

end largest_four_digit_number_with_conditions_l2169_216989


namespace power_exceeds_million_l2169_216943

theorem power_exceeds_million : ∃ (n₁ n₂ n₃ : ℕ+),
  (1.01 : ℝ) ^ (n₁ : ℕ) > 1000000 ∧
  (1.001 : ℝ) ^ (n₂ : ℕ) > 1000000 ∧
  (1.000001 : ℝ) ^ (n₃ : ℕ) > 1000000 := by
  sorry

end power_exceeds_million_l2169_216943


namespace apple_distribution_l2169_216934

theorem apple_distribution (x : ℕ) (total_apples : ℕ) : 
  (total_apples = 5 * x + 12) → 
  (total_apples < 8 * x) →
  (0 ≤ 5 * x + 12 - 8 * (x - 1) ∧ 5 * x + 12 - 8 * (x - 1) < 8) :=
by sorry

end apple_distribution_l2169_216934


namespace intersection_of_three_lines_l2169_216988

/-- If three lines intersect at one point, then a specific value of a is determined -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃! p : ℝ × ℝ, (a * p.1 + 2 * p.2 + 8 = 0) ∧ 
                  (4 * p.1 + 3 * p.2 - 10 = 0) ∧ 
                  (2 * p.1 - p.2 = 0)) → 
  a = -12 := by
  sorry

end intersection_of_three_lines_l2169_216988


namespace problem_2023_l2169_216967

theorem problem_2023 : (2023^2 - 2023 + 1) / 2023 = 2022 + 1/2023 := by
  sorry

end problem_2023_l2169_216967


namespace sequence_exceeds_1994_l2169_216915

/-- A sequence satisfying the given conditions -/
def SpecialSequence (x : ℕ → ℝ) (k : ℝ) : Prop :=
  (x 0 = 1) ∧
  (x 1 = 1 + k) ∧
  (k > 0) ∧
  (∀ n, x (2*n + 1) - x (2*n) = x (2*n) - x (2*n - 1)) ∧
  (∀ n, x (2*n) / x (2*n - 1) = x (2*n - 1) / x (2*n - 2))

/-- The main theorem stating that the sequence eventually exceeds 1994 -/
theorem sequence_exceeds_1994 {x : ℕ → ℝ} {k : ℝ} (h : SpecialSequence x k) :
  ∃ N, ∀ n ≥ N, x n > 1994 :=
sorry

end sequence_exceeds_1994_l2169_216915


namespace no_solution_iff_k_equals_four_l2169_216926

theorem no_solution_iff_k_equals_four :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 4 := by
  sorry

end no_solution_iff_k_equals_four_l2169_216926


namespace equality_of_solution_sets_implies_sum_l2169_216908

theorem equality_of_solution_sets_implies_sum (a b : ℝ) : 
  (∀ x : ℝ, |x - 2| > 1 ↔ x^2 + a*x + b > 0) → a + b = -1 := by
  sorry

end equality_of_solution_sets_implies_sum_l2169_216908


namespace chocolate_eggs_weight_l2169_216904

/-- Calculates the total weight of remaining chocolate eggs after one box is discarded -/
theorem chocolate_eggs_weight (total_eggs : ℕ) (egg_weight : ℕ) (num_boxes : ℕ) :
  total_eggs = 12 →
  egg_weight = 10 →
  num_boxes = 4 →
  (total_eggs * egg_weight) - (total_eggs / num_boxes * egg_weight) = 90 :=
by
  sorry

end chocolate_eggs_weight_l2169_216904


namespace absolute_value_inequality_l2169_216958

theorem absolute_value_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ x ∈ Set.Icc (-2) 1 ∪ Set.Icc 5 8 :=
sorry

end absolute_value_inequality_l2169_216958


namespace least_months_to_double_debt_l2169_216932

def initial_amount : ℝ := 1500
def monthly_rate : ℝ := 0.06

def amount_owed (t : ℕ) : ℝ :=
  initial_amount * (1 + monthly_rate) ^ t

theorem least_months_to_double_debt :
  (∀ n < 12, amount_owed n ≤ 2 * initial_amount) ∧
  amount_owed 12 > 2 * initial_amount :=
sorry

end least_months_to_double_debt_l2169_216932


namespace sufficient_but_not_necessary_conditions_l2169_216951

theorem sufficient_but_not_necessary_conditions (a b : ℝ) :
  (∀ (a b : ℝ), a + b > 2 → a + b > 0) ∧
  (∀ (a b : ℝ), (a > 0 ∧ b > 0) → a + b > 0) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a + b > 2)) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a > 0 ∧ b > 0)) :=
by sorry


end sufficient_but_not_necessary_conditions_l2169_216951


namespace square_of_103_product_of_998_and_1002_l2169_216955

-- Problem 1
theorem square_of_103 : 103^2 = 10609 := by sorry

-- Problem 2
theorem product_of_998_and_1002 : 998 * 1002 = 999996 := by sorry

end square_of_103_product_of_998_and_1002_l2169_216955


namespace problem_statement_l2169_216984

theorem problem_statement (a : ℝ) (h : 2 * a - 1 / a = 3) : 16 * a^4 + 1 / a^4 = 161 := by
  sorry

end problem_statement_l2169_216984


namespace flagpole_height_l2169_216918

-- Define the given conditions
def flagpoleShadowLength : ℝ := 45
def buildingShadowLength : ℝ := 65
def buildingHeight : ℝ := 26

-- Define the theorem
theorem flagpole_height :
  ∃ (h : ℝ), h / flagpoleShadowLength = buildingHeight / buildingShadowLength ∧ h = 18 := by
  sorry

end flagpole_height_l2169_216918


namespace complex_equation_difference_l2169_216930

theorem complex_equation_difference (x y : ℝ) : 
  (x : ℂ) + y * I = 1 + 2 * x * I → x - y = -1 := by
  sorry

end complex_equation_difference_l2169_216930


namespace price_reduction_effect_l2169_216957

theorem price_reduction_effect (original_price : ℝ) (original_sales : ℝ) 
  (price_reduction_percent : ℝ) (net_effect_percent : ℝ) : 
  price_reduction_percent = 40 →
  net_effect_percent = 8 →
  ∃ (sales_increase_percent : ℝ),
    sales_increase_percent = 80 ∧
    (1 - price_reduction_percent / 100) * (1 + sales_increase_percent / 100) = 1 + net_effect_percent / 100 :=
by sorry

end price_reduction_effect_l2169_216957


namespace raisin_cookies_sold_l2169_216968

theorem raisin_cookies_sold (raisin oatmeal : ℕ) : 
  (raisin : ℚ) / oatmeal = 6 / 1 →
  raisin + oatmeal = 49 →
  raisin = 42 := by
sorry

end raisin_cookies_sold_l2169_216968


namespace alcohol_solution_proof_l2169_216947

/-- Proves that adding a specific amount of pure alcohol to a given solution results in the desired alcohol percentage -/
theorem alcohol_solution_proof (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) (final_percentage : ℝ) :
  initial_volume = 100 →
  initial_percentage = 0.2 →
  added_alcohol = 14.285714285714286 →
  final_percentage = 0.3 →
  (initial_volume * initial_percentage + added_alcohol) / (initial_volume + added_alcohol) = final_percentage := by
  sorry

#check alcohol_solution_proof

end alcohol_solution_proof_l2169_216947


namespace people_not_buying_coffee_l2169_216912

theorem people_not_buying_coffee (total_people : ℕ) (coffee_ratio : ℚ) 
  (h1 : total_people = 25) 
  (h2 : coffee_ratio = 3/5) : 
  total_people - (coffee_ratio * total_people).floor = 10 := by
  sorry

end people_not_buying_coffee_l2169_216912


namespace square_root_equality_l2169_216993

theorem square_root_equality (x a : ℝ) (hx : x > 0) :
  (Real.sqrt x = 2 * a - 1 ∧ Real.sqrt x = -a + 2) → x = 9 := by
  sorry

end square_root_equality_l2169_216993


namespace equal_angles_same_terminal_side_l2169_216954

/-- Represents an angle in the coordinate system -/
structure Angle where
  value : ℝ

/-- Represents the terminal side of an angle -/
structure TerminalSide where
  x : ℝ
  y : ℝ

/-- Returns the terminal side of an angle -/
noncomputable def terminalSide (a : Angle) : TerminalSide :=
  { x := Real.cos a.value, y := Real.sin a.value }

/-- Theorem: Equal angles have the same terminal side -/
theorem equal_angles_same_terminal_side (a b : Angle) :
  a = b → terminalSide a = terminalSide b := by
  sorry

end equal_angles_same_terminal_side_l2169_216954


namespace quadratic_inequality_solution_l2169_216999

-- Define the quadratic inequality
def quadratic_inequality (x : ℝ) : Prop := (x - 2) * (x + 2) < 5

-- Define the solution set
def solution_set : Set ℝ := {x | -3 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | quadratic_inequality x} = solution_set :=
sorry

end quadratic_inequality_solution_l2169_216999


namespace initial_typists_count_initial_typists_count_proof_l2169_216901

/-- Given that some typists can type 38 letters in 20 minutes and 30 typists working at the same rate can complete 171 letters in 1 hour, prove that the number of typists in the initial group is 20. -/
theorem initial_typists_count : ℕ :=
  let initial_letters : ℕ := 38
  let initial_time : ℕ := 20
  let second_typists : ℕ := 30
  let second_letters : ℕ := 171
  let second_time : ℕ := 60
  20

/-- Proof of the theorem -/
theorem initial_typists_count_proof : initial_typists_count = 20 := by
  sorry

end initial_typists_count_initial_typists_count_proof_l2169_216901


namespace quadratic_one_root_l2169_216982

/-- A quadratic function with coefficients a, b, and c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The discriminant of a quadratic function -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_one_root (k : ℝ) : 
  (∃! x, QuadraticFunction 1 (-2) k x = 0) → k = 1 := by
  sorry

end quadratic_one_root_l2169_216982


namespace power_of_two_equality_l2169_216986

theorem power_of_two_equality : (2^36 / 8 = 2^x) → x = 33 := by
  sorry

end power_of_two_equality_l2169_216986


namespace share_of_A_l2169_216939

theorem share_of_A (total : ℚ) (a b c : ℚ) : 
  total = 510 →
  a = (2 / 3) * b →
  b = (1 / 4) * c →
  total = a + b + c →
  a = 60 := by
sorry

end share_of_A_l2169_216939


namespace frisbee_price_problem_l2169_216925

theorem frisbee_price_problem (total_frisbees : ℕ) (total_revenue : ℕ) 
  (price_some : ℕ) (min_sold_at_price_some : ℕ) :
  total_frisbees = 64 →
  total_revenue = 200 →
  price_some = 4 →
  min_sold_at_price_some = 8 →
  ∃ (price_others : ℕ), 
    price_others = 3 ∧
    ∃ (num_at_price_some : ℕ),
      num_at_price_some ≥ min_sold_at_price_some ∧
      price_some * num_at_price_some + price_others * (total_frisbees - num_at_price_some) = total_revenue :=
by sorry

end frisbee_price_problem_l2169_216925


namespace zoe_spent_30_dollars_l2169_216922

/-- The price of a single flower in dollars -/
def flower_price : ℕ := 3

/-- The number of roses Zoe bought -/
def roses_bought : ℕ := 8

/-- The number of daisies Zoe bought -/
def daisies_bought : ℕ := 2

/-- Theorem: Given the conditions, Zoe spent 30 dollars -/
theorem zoe_spent_30_dollars : 
  (roses_bought + daisies_bought) * flower_price = 30 := by
  sorry

end zoe_spent_30_dollars_l2169_216922


namespace max_d_value_l2169_216903

def a (n : ℕ+) : ℕ := 80 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ n : ℕ+, d n = 5) ∧ (∀ n : ℕ+, d n ≤ 5) :=
sorry

end max_d_value_l2169_216903


namespace inradius_value_l2169_216913

/-- Given a triangle with perimeter p and area A, its inradius r satisfies A = r * p / 2 -/
axiom inradius_formula (p A r : ℝ) : A = r * p / 2

/-- The perimeter of the triangle -/
def p : ℝ := 42

/-- The area of the triangle -/
def A : ℝ := 105

/-- The inradius of the triangle -/
def r : ℝ := 5

theorem inradius_value : r = 5 := by sorry

end inradius_value_l2169_216913


namespace kindergarten_allergies_l2169_216987

/-- Given a kindergarten with the following conditions:
  - T is the total number of children
  - Half of the children are allergic to peanuts
  - 10 children are not allergic to cashew nuts
  - 10 children are allergic to both peanuts and cashew nuts
  - Some children are allergic to cashew nuts
Prove that the number of children not allergic to peanuts and not allergic to cashew nuts is 10 -/
theorem kindergarten_allergies (T : ℕ) : 
  T > 0 →
  T / 2 = (T - T / 2) → -- Half of the children are allergic to peanuts
  ∃ (cashew_allergic : ℕ), cashew_allergic > 0 ∧ cashew_allergic < T → -- Some children are allergic to cashew nuts
  10 = T - cashew_allergic → -- 10 children are not allergic to cashew nuts
  10 ≤ T / 2 → -- 10 children are allergic to both peanuts and cashew nuts
  10 = T - (T / 2 + cashew_allergic - 10) -- Number of children not allergic to peanuts and not allergic to cashew nuts
  := by sorry

end kindergarten_allergies_l2169_216987


namespace distance_point_to_line_is_correct_l2169_216985

def point_A : ℝ × ℝ × ℝ := (0, 3, -1)
def point_B : ℝ × ℝ × ℝ := (1, 2, 1)
def point_C : ℝ × ℝ × ℝ := (2, 4, 0)

def line_direction : ℝ × ℝ × ℝ := (point_C.1 - point_B.1, point_C.2.1 - point_B.2.1, point_C.2.2 - point_B.2.2)

def distance_point_to_line (A B C : ℝ × ℝ × ℝ) : ℝ := sorry

theorem distance_point_to_line_is_correct :
  distance_point_to_line point_A point_B point_C = (3 * Real.sqrt 2) / 2 := by sorry

end distance_point_to_line_is_correct_l2169_216985


namespace product_local_abs_value_l2169_216914

/-- The local value of a digit in a number -/
def localValue (n : ℕ) (d : ℕ) (p : ℕ) : ℕ := d * (10 ^ p)

/-- The absolute value of a natural number -/
def absValue (n : ℕ) : ℕ := n

/-- The given number -/
def givenNumber : ℕ := 564823

/-- The digit we're focusing on -/
def focusDigit : ℕ := 4

/-- The position of the focus digit (0-indexed from right) -/
def digitPosition : ℕ := 4

theorem product_local_abs_value : 
  localValue givenNumber focusDigit digitPosition * absValue focusDigit = 160000 := by
  sorry

end product_local_abs_value_l2169_216914


namespace slope_at_five_is_zero_l2169_216983

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem slope_at_five_is_zero
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_diff : Differentiable ℝ f)
  (h_period : has_period f 5) :
  deriv f 5 = 0 := by
  sorry

end slope_at_five_is_zero_l2169_216983


namespace isosceles_trapezoid_area_l2169_216991

/-- The area of an isosceles trapezoid with given dimensions -/
theorem isosceles_trapezoid_area (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let leg := a
  let base1 := b
  let base2 := c
  let height := Real.sqrt (a^2 - ((c - b)/2)^2)
  (base1 + base2) * height / 2 = 36 ↔ a = 5 ∧ b = 6 ∧ c = 12 :=
by sorry

end isosceles_trapezoid_area_l2169_216991


namespace most_likely_white_balls_l2169_216975

/-- Represents a box of balls -/
structure BallBox where
  total : ℕ
  white : ℕ
  black : ℕ
  white_le_total : white ≤ total
  black_eq_total_sub_white : black = total - white

/-- Represents the result of multiple draws -/
structure DrawResult where
  total_draws : ℕ
  white_draws : ℕ
  white_draws_le_total : white_draws ≤ total_draws

/-- The probability of drawing a white ball given a box configuration -/
def draw_probability (box : BallBox) : ℚ :=
  box.white / box.total

/-- The likelihood of a draw result given a box configuration -/
def draw_likelihood (box : BallBox) (result : DrawResult) : ℚ :=
  (draw_probability box) ^ result.white_draws * (1 - draw_probability box) ^ (result.total_draws - result.white_draws)

/-- Theorem: Given 10 balls and 240 white draws out of 400, 6 white balls is most likely -/
theorem most_likely_white_balls 
  (box : BallBox) 
  (result : DrawResult) 
  (h_total : box.total = 10) 
  (h_draws : result.total_draws = 400) 
  (h_white_draws : result.white_draws = 240) :
  (∀ (other_box : BallBox), other_box.total = 10 → 
    draw_likelihood box result ≥ draw_likelihood other_box result) →
  box.white = 6 :=
sorry

end most_likely_white_balls_l2169_216975


namespace inverse_proportion_quadrants_l2169_216900

/-- An inverse proportion function passing through (3, -2) lies in the second and fourth quadrants -/
theorem inverse_proportion_quadrants :
  ∀ (k : ℝ), k ≠ 0 →
  (∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = k / x) ∧ f 3 = -2) →
  (∀ x y, (x > 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0)) :=
by sorry

end inverse_proportion_quadrants_l2169_216900


namespace rectangles_and_triangles_on_4x3_grid_l2169_216998

/-- The number of rectangles on an m × n grid -/
def count_rectangles (m n : ℕ) : ℕ := (m.choose 2) * (n.choose 2)

/-- The number of right-angled triangles (with right angles at grid points) on an m × n grid -/
def count_right_triangles (m n : ℕ) : ℕ := 2 * (m - 1) * (n - 1)

/-- The total number of rectangles and right-angled triangles on a 4×3 grid is 30 -/
theorem rectangles_and_triangles_on_4x3_grid :
  count_rectangles 4 3 + count_right_triangles 4 3 = 30 := by
  sorry

end rectangles_and_triangles_on_4x3_grid_l2169_216998


namespace problem_solution_l2169_216963

theorem problem_solution (x z : ℝ) (h1 : x ≠ 0) (h2 : x/3 = z^2) (h3 : x/5 = 5*z) : x = 625/3 := by
  sorry

end problem_solution_l2169_216963


namespace inequality_proof_l2169_216956

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z ≥ 3) :
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 ∧
  ((1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end inequality_proof_l2169_216956


namespace angle_measure_proof_l2169_216973

theorem angle_measure_proof (x : ℝ) : 
  (x = 21) → 
  (90 - x = 3 * x + 6) ∧
  (x + (90 - x) = 90) :=
by sorry

end angle_measure_proof_l2169_216973


namespace jake_alcohol_consumption_l2169_216927

-- Define the given constants
def total_shots : ℚ := 8
def ounces_per_shot : ℚ := 3/2
def alcohol_percentage : ℚ := 1/2

-- Define Jake's share of shots
def jakes_shots : ℚ := total_shots / 2

-- Define the function to calculate pure alcohol consumed
def pure_alcohol_consumed : ℚ :=
  jakes_shots * ounces_per_shot * alcohol_percentage

-- Theorem statement
theorem jake_alcohol_consumption :
  pure_alcohol_consumed = 3 := by sorry

end jake_alcohol_consumption_l2169_216927


namespace base_seven_23456_equals_6068_l2169_216990

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base_seven_23456_equals_6068 :
  base_seven_to_ten [6, 5, 4, 3, 2] = 6068 := by
  sorry

end base_seven_23456_equals_6068_l2169_216990


namespace units_digit_sum_factorials_30_l2169_216978

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_30 :
  units_digit (sum_factorials 30) = units_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4) :=
by sorry

end units_digit_sum_factorials_30_l2169_216978


namespace equation_solutions_l2169_216940

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃! x : ℝ, a * x + b = 0) ∨ (∀ x : ℝ, a * x + b = 0) :=
sorry

end equation_solutions_l2169_216940


namespace smallest_value_for_x_between_0_and_1_l2169_216996

theorem smallest_value_for_x_between_0_and_1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 ≤ x ∧ x^3 ≤ x^2 ∧ x^3 ≤ x^3 ∧ x^3 ≤ Real.sqrt x ∧ x^3 ≤ 2*x ∧ x^3 ≤ 1/x :=
by sorry

end smallest_value_for_x_between_0_and_1_l2169_216996


namespace n_has_9_digits_l2169_216933

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_minimal : ∀ m : ℕ, m > 0 → 30 ∣ m → (∃ k : ℕ, m^2 = k^3) → (∃ k : ℕ, m^3 = k^2) → n ≤ m

/-- The number of digits in n -/
def num_digits (x : ℕ) : ℕ := sorry

/-- Theorem stating that n has 9 digits -/
theorem n_has_9_digits : num_digits n = 9 := by sorry

end n_has_9_digits_l2169_216933


namespace mixed_nuts_cost_l2169_216974

/-- Represents the price and amount of a type of nut -/
structure NutInfo where
  price : ℚ  -- Price in dollars
  amount : ℚ  -- Amount in ounces
  deriving Repr

/-- Calculates the discounted price per ounce -/
def discountedPricePerOz (info : NutInfo) (discount : ℚ) : ℚ :=
  (info.price / info.amount) * (1 - discount)

/-- Calculates the cost of a nut in the mix -/
def nutCostInMix (pricePerOz : ℚ) (proportion : ℚ) : ℚ :=
  pricePerOz * proportion

/-- The main theorem stating the minimum cost of the mixed nuts -/
theorem mixed_nuts_cost
  (almond_info : NutInfo)
  (cashew_info : NutInfo)
  (walnut_info : NutInfo)
  (almond_discount cashew_discount walnut_discount : ℚ)
  (h_almond_price : almond_info.price = 18)
  (h_almond_amount : almond_info.amount = 32)
  (h_cashew_price : cashew_info.price = 45/2)
  (h_cashew_amount : cashew_info.amount = 28)
  (h_walnut_price : walnut_info.price = 15)
  (h_walnut_amount : walnut_info.amount = 24)
  (h_almond_discount : almond_discount = 1/10)
  (h_cashew_discount : cashew_discount = 3/20)
  (h_walnut_discount : walnut_discount = 1/5)
  : ∃ (cost : ℕ), cost = 56 ∧ 
    cost * (1/100 : ℚ) ≥ 
      nutCostInMix (discountedPricePerOz almond_info almond_discount) (1/2) +
      nutCostInMix (discountedPricePerOz cashew_info cashew_discount) (3/10) +
      nutCostInMix (discountedPricePerOz walnut_info walnut_discount) (1/5) :=
sorry

end mixed_nuts_cost_l2169_216974


namespace total_limes_picked_l2169_216950

theorem total_limes_picked (fred_limes alyssa_limes nancy_limes david_limes eileen_limes : ℕ)
  (h1 : fred_limes = 36)
  (h2 : alyssa_limes = 32)
  (h3 : nancy_limes = 35)
  (h4 : david_limes = 42)
  (h5 : eileen_limes = 50) :
  fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes = 195 := by
  sorry

end total_limes_picked_l2169_216950
