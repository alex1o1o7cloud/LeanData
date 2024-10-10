import Mathlib

namespace f_composition_value_l1922_192299

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_value : f (f (f (-1))) = Real.pi + 1 := by sorry

end f_composition_value_l1922_192299


namespace parallelogram_cross_section_exists_l1922_192290

/-- A cuboid in 3D space -/
structure Cuboid where
  -- Define the cuboid structure (you may need to add more fields)
  dummy : Unit

/-- A plane in 3D space -/
structure Plane where
  -- Define the plane structure (you may need to add more fields)
  dummy : Unit

/-- The cross-section resulting from a plane intersecting a cuboid -/
def crossSection (c : Cuboid) (p : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry -- Define the cross-section

/-- A predicate to check if a set of points forms a parallelogram -/
def isParallelogram (s : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry -- Define the conditions for a parallelogram

/-- Theorem stating that there exists a plane that intersects a cuboid to form a parallelogram cross-section -/
theorem parallelogram_cross_section_exists :
  ∃ (c : Cuboid) (p : Plane), isParallelogram (crossSection c p) :=
sorry

end parallelogram_cross_section_exists_l1922_192290


namespace cos_2alpha_2beta_l1922_192262

theorem cos_2alpha_2beta (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
  sorry

end cos_2alpha_2beta_l1922_192262


namespace range_of_m_l1922_192223

-- Define the sets A and B
def A : Set ℝ := {x | (2 - x) / (2 * x - 1) > 1}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*x + 1 - m ≤ 0}

-- State the theorem
theorem range_of_m (h : ∀ m > 0, A ⊆ B m ∧ ∃ x, x ∈ B m ∧ x ∉ A) :
  {m : ℝ | m ≥ 4} = {m : ℝ | m > 0 ∧ A ⊆ B m ∧ ∃ x, x ∈ B m ∧ x ∉ A} :=
by sorry

end range_of_m_l1922_192223


namespace tree_growth_theorem_l1922_192238

-- Define growth rates and initial heights
def growth_rate_A : ℚ := 25  -- 50 cm / 2 weeks
def growth_rate_B : ℚ := 70 / 3
def growth_rate_C : ℚ := 90 / 4
def initial_height_A : ℚ := 200
def initial_height_B : ℚ := 150
def initial_height_C : ℚ := 250
def weeks : ℕ := 16

-- Calculate final heights
def final_height_A : ℚ := initial_height_A + growth_rate_A * weeks
def final_height_B : ℚ := initial_height_B + growth_rate_B * weeks
def final_height_C : ℚ := initial_height_C + growth_rate_C * weeks

-- Define the combined final height
def combined_final_height : ℚ := final_height_A + final_height_B + final_height_C

-- Theorem to prove
theorem tree_growth_theorem :
  (combined_final_height : ℚ) = 1733.33 := by
  sorry

end tree_growth_theorem_l1922_192238


namespace quadratic_root_interval_l1922_192208

theorem quadratic_root_interval (a b : ℝ) (hb : b > 0) 
  (h_distinct : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a*x₁ + b = 0 ∧ x₂^2 + a*x₂ + b = 0)
  (h_one_in_unit : ∃! x : ℝ, x^2 + a*x + b = 0 ∧ x ∈ Set.Icc (-1) 1) :
  ∃! x : ℝ, x^2 + a*x + b = 0 ∧ x ∈ Set.Ioo (-b) b :=
sorry

end quadratic_root_interval_l1922_192208


namespace rocky_training_ratio_l1922_192245

/-- Rocky's training schedule over three days -/
structure TrainingSchedule where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- Conditions for Rocky's training -/
def validSchedule (s : TrainingSchedule) : Prop :=
  s.day1 = 4 ∧ 
  s.day2 = 2 * s.day1 ∧ 
  s.day3 > s.day2 ∧
  s.day1 + s.day2 + s.day3 = 36

/-- The ratio of miles run on day 3 to day 2 is 3 -/
theorem rocky_training_ratio (s : TrainingSchedule) 
  (h : validSchedule s) : s.day3 / s.day2 = 3 := by
  sorry


end rocky_training_ratio_l1922_192245


namespace square_difference_equality_l1922_192266

theorem square_difference_equality : 1004^2 - 996^2 - 1002^2 + 998^2 = 8000 := by
  sorry

end square_difference_equality_l1922_192266


namespace range_of_G_l1922_192276

-- Define the function G
def G (x : ℝ) : ℝ := |x + 2| - 2 * |x - 2|

-- State the theorem about the range of G
theorem range_of_G :
  Set.range G = Set.Icc (-8 : ℝ) 8 :=
sorry

end range_of_G_l1922_192276


namespace problem_solution_l1922_192213

def vector := ℝ × ℝ

noncomputable def problem (x : ℝ) : Prop :=
  let a : vector := (1, Real.sin x)
  let b : vector := (Real.sin x, -1)
  let c : vector := (1, Real.cos x)
  0 < x ∧ x < Real.pi ∧
  ¬ (∃ (k : ℝ), (1 + Real.sin x, Real.sin x - 1) = (k * c.1, k * c.2)) ∧
  x = Real.pi / 2 ∧
  ∃ (A B C : ℝ), 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    B = Real.pi / 2 ∧
    2 * (Real.sin B)^2 + 2 * (Real.sin C)^2 - 2 * (Real.sin A)^2 = Real.sin B * Real.sin C

theorem problem_solution (x : ℝ) (h : problem x) :
  ∃ (A B C : ℝ), Real.sin (C - Real.pi / 3) = (1 - 3 * Real.sqrt 5) / 8 := by
  sorry

#check problem_solution

end problem_solution_l1922_192213


namespace fraction_simplification_l1922_192200

theorem fraction_simplification :
  (1/2 - 1/3 + 1/5) / (1/3 - 1/2 + 1/7) = -77/5 := by
  sorry

end fraction_simplification_l1922_192200


namespace tiles_per_row_l1922_192284

/-- Proves that a square room with an area of 144 square feet,
    when covered with 8-inch by 8-inch tiles, will have 18 tiles in each row. -/
theorem tiles_per_row (room_area : ℝ) (tile_size : ℝ) :
  room_area = 144 →
  tile_size = 8 →
  (Real.sqrt room_area * 12) / tile_size = 18 := by
  sorry

end tiles_per_row_l1922_192284


namespace land_conversion_equation_l1922_192234

/-- Represents the land conversion scenario in a village --/
theorem land_conversion_equation (x : ℝ) : 
  (54 - x = (20 / 100) * (108 + x)) ↔ 
  (54 - x = 0.2 * (108 + x) ∧ 
   0 ≤ x ∧ 
   x ≤ 54 ∧
   108 + x > 0) := by
  sorry

end land_conversion_equation_l1922_192234


namespace wafting_pie_egg_usage_l1922_192210

/-- The Wafting Pie Company's egg usage problem -/
theorem wafting_pie_egg_usage 
  (total_eggs : ℕ) 
  (morning_eggs : ℕ) 
  (h1 : total_eggs = 1339)
  (h2 : morning_eggs = 816) :
  total_eggs - morning_eggs = 523 := by
  sorry

end wafting_pie_egg_usage_l1922_192210


namespace orchestra_members_count_l1922_192293

theorem orchestra_members_count : ∃! n : ℕ,
  150 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 1 ∧
  n % 8 = 3 ∧
  n % 9 = 5 ∧
  n = 211 := by sorry

end orchestra_members_count_l1922_192293


namespace quadratic_roots_l1922_192259

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 2 ∧ 
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_roots_l1922_192259


namespace apple_boxes_bought_l1922_192215

-- Define the variables
variable (cherry_price : ℝ) -- Price of one cherry
variable (apple_price : ℝ) -- Price of one apple
variable (cherry_size : ℝ) -- Size of one cherry
variable (apple_size : ℝ) -- Size of one apple
variable (cherries_per_box : ℕ) -- Number of cherries in a box

-- Define the conditions
axiom price_relation : 2 * cherry_price = 3 * apple_price
axiom size_relation : apple_size = 12 * cherry_size
axiom box_size_equality : cherries_per_box * cherry_size = cherries_per_box * apple_size

-- Define the theorem
theorem apple_boxes_bought (h : cherries_per_box > 0) :
  (cherries_per_box * cherry_price) / apple_price = 18 := by
  sorry

end apple_boxes_bought_l1922_192215


namespace card_drawing_probability_l1922_192230

/-- Represents a standard 52-card deck --/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in each suit --/
def CardsPerSuit : ℕ := 13

/-- Represents the number of suits in a standard deck --/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards drawn --/
def CardsDrawn : ℕ := 8

/-- The probability of the specified event occurring --/
def probability_of_event : ℚ := 3 / 16384

theorem card_drawing_probability :
  (1 : ℚ) / NumberOfSuits *     -- Probability of first card being any suit
  (3 : ℚ) / NumberOfSuits *     -- Probability of second card being a different suit
  (2 : ℚ) / NumberOfSuits *     -- Probability of third card being a different suit
  (1 : ℚ) / NumberOfSuits *     -- Probability of fourth card being the remaining suit
  ((1 : ℚ) / NumberOfSuits)^4   -- Probability of next four cards matching the suit sequence
  = probability_of_event := by sorry

#check card_drawing_probability

end card_drawing_probability_l1922_192230


namespace cubic_polynomial_root_l1922_192260

theorem cubic_polynomial_root (a b : ℚ) :
  (∃ (x : ℝ), x^3 + a*x + b = 0 ∧ x = 3 - Real.sqrt 5) →
  (∃ (r : ℤ), r^3 + a*r + b = 0) →
  (∃ (r : ℤ), r^3 + a*r + b = 0 ∧ r = 0) :=
by sorry

end cubic_polynomial_root_l1922_192260


namespace necessary_but_not_sufficient_l1922_192289

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 = 3*x + 4) → (x = Real.sqrt (3*x + 4)) ∧
  ¬(∀ x : ℝ, (x = Real.sqrt (3*x + 4)) → (x^2 = 3*x + 4)) :=
by sorry

end necessary_but_not_sufficient_l1922_192289


namespace jack_closet_capacity_l1922_192201

/-- Represents the storage capacity of a closet -/
structure ClosetCapacity where
  cansPerRow : ℕ
  rowsPerShelf : ℕ
  shelvesPerCloset : ℕ

/-- Calculates the total number of cans that can be stored in a closet -/
def totalCansPerCloset (c : ClosetCapacity) : ℕ :=
  c.cansPerRow * c.rowsPerShelf * c.shelvesPerCloset

/-- Theorem: Given Jack's closet configuration, he can store 480 cans in each closet -/
theorem jack_closet_capacity :
  let jackCloset : ClosetCapacity := {
    cansPerRow := 12,
    rowsPerShelf := 4,
    shelvesPerCloset := 10
  }
  totalCansPerCloset jackCloset = 480 := by
  sorry


end jack_closet_capacity_l1922_192201


namespace set_operations_and_range_l1922_192277

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a - 1}

-- State the theorem
theorem set_operations_and_range :
  (A ∩ B = {x : ℝ | 3 < x ∧ x < 6}) ∧
  ((Set.univ \ A) ∪ (Set.univ \ B) = {x : ℝ | x ≤ 3 ∨ x ≥ 6}) ∧
  (∀ a : ℝ, (B ∪ C a = B) ↔ (a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5))) :=
by sorry

end set_operations_and_range_l1922_192277


namespace trigonometric_expression_evaluation_l1922_192270

theorem trigonometric_expression_evaluation :
  (Real.sin (20 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 1/3 := by
sorry

end trigonometric_expression_evaluation_l1922_192270


namespace count_common_divisors_84_90_l1922_192220

def common_divisors (a b : ℕ) : Finset ℕ :=
  (Finset.range (min a b + 1)).filter (fun d => d > 1 ∧ a % d = 0 ∧ b % d = 0)

theorem count_common_divisors_84_90 :
  (common_divisors 84 90).card = 3 := by
  sorry

end count_common_divisors_84_90_l1922_192220


namespace equation_c_is_quadratic_l1922_192265

/-- A quadratic equation in one variable is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing (x-1)(x+2)=1 --/
def f (x : ℝ) : ℝ := (x - 1) * (x + 2) - 1

theorem equation_c_is_quadratic : is_quadratic_equation f := by
  sorry

end equation_c_is_quadratic_l1922_192265


namespace remainder_of_m_l1922_192224

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end remainder_of_m_l1922_192224


namespace sum_division_l1922_192247

/-- The problem of dividing a sum among x, y, and z -/
theorem sum_division (x y z : ℝ) : 
  (∀ (r : ℝ), y = 0.45 * r → z = 0.5 * r → x = r) →  -- For each rupee x gets, y gets 0.45 and z gets 0.5
  y = 63 →  -- y's share is 63 rupees
  x + y + z = 273 := by  -- The total amount is 273 rupees
sorry


end sum_division_l1922_192247


namespace function_value_problem_l1922_192295

theorem function_value_problem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (2 * x + 1) = 3 * x - 2) →
  f a = 4 →
  a = 5 := by
sorry

end function_value_problem_l1922_192295


namespace steve_wood_needed_l1922_192248

/-- The amount of wood Steve needs to buy for his bench project -/
def total_wood_needed (long_pieces : ℕ) (long_length : ℕ) (short_pieces : ℕ) (short_length : ℕ) : ℕ :=
  long_pieces * long_length + short_pieces * short_length

/-- Proof that Steve needs to buy 28 feet of wood -/
theorem steve_wood_needed : total_wood_needed 6 4 2 2 = 28 := by
  sorry

end steve_wood_needed_l1922_192248


namespace no_square_cut_with_250_remaining_l1922_192272

theorem no_square_cut_with_250_remaining : ¬∃ (n m : ℕ), n > m ∧ n^2 - m^2 = 250 := by
  sorry

end no_square_cut_with_250_remaining_l1922_192272


namespace odd_even_functions_inequality_l1922_192205

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_functions_inequality (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g)
  (h_diff : ∀ x, f x - g x = (1/2)^x) :
  g 1 < f 0 ∧ f 0 < f (-1) := by
  sorry

end odd_even_functions_inequality_l1922_192205


namespace nancy_apples_l1922_192282

def mike_apples : ℕ := 7
def keith_apples : ℕ := 6
def total_apples : ℕ := 16

theorem nancy_apples :
  total_apples - (mike_apples + keith_apples) = 3 :=
by sorry

end nancy_apples_l1922_192282


namespace fly_path_shortest_distance_l1922_192229

/-- Represents a right circular cone. -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone. -/
structure SurfacePoint where
  distanceFromVertex : ℝ

/-- Calculates the shortest distance between two points on the surface of a cone. -/
def shortestSurfaceDistance (c : Cone) (p1 p2 : SurfacePoint) : ℝ :=
  sorry

theorem fly_path_shortest_distance :
  let c : Cone := { baseRadius := 600, height := 200 * Real.sqrt 7 }
  let p1 : SurfacePoint := { distanceFromVertex := 125 }
  let p2 : SurfacePoint := { distanceFromVertex := 375 * Real.sqrt 2 }
  shortestSurfaceDistance c p1 p2 = 625 := by sorry

end fly_path_shortest_distance_l1922_192229


namespace puppies_given_away_l1922_192228

/-- Given that Sandy initially had some puppies and now has fewer,
    prove that the number of puppies given away is the difference
    between the initial and current number of puppies. -/
theorem puppies_given_away
  (initial_puppies : ℕ)
  (current_puppies : ℕ)
  (h : current_puppies ≤ initial_puppies) :
  initial_puppies - current_puppies =
  initial_puppies - current_puppies :=
by sorry

end puppies_given_away_l1922_192228


namespace intersection_of_A_and_B_l1922_192263

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x < 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l1922_192263


namespace circle_condition_tangent_circles_intersecting_circle_line_l1922_192241

-- Define the equation C
def equation_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the given circle equation
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 12*y + 36 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem 1: For equation C to represent a circle, m < 5
theorem circle_condition (m : ℝ) : 
  (∃ x y, equation_C x y m) → m < 5 :=
sorry

-- Theorem 2: When circle C is tangent to the given circle, m = 4
theorem tangent_circles (m : ℝ) :
  (∃ x y, equation_C x y m ∧ given_circle x y) → m = 4 :=
sorry

-- Theorem 3: When circle C intersects line l at points M and N with |MN| = 4√5/5, m = 4
theorem intersecting_circle_line (m : ℝ) :
  (∃ x1 y1 x2 y2, 
    equation_C x1 y1 m ∧ equation_C x2 y2 m ∧
    line_l x1 y1 ∧ line_l x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = (4*Real.sqrt 5/5)^2) → 
  m = 4 :=
sorry

end circle_condition_tangent_circles_intersecting_circle_line_l1922_192241


namespace lemon_sequences_l1922_192209

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of times the class meets in a week -/
def meetings_per_week : ℕ := 5

/-- The number of possible sequences of lemon recipients in a week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

/-- Theorem stating the number of possible sequences of lemon recipients -/
theorem lemon_sequences :
  num_sequences = 759375 :=
by sorry

end lemon_sequences_l1922_192209


namespace two_person_subcommittees_with_male_l1922_192225

theorem two_person_subcommittees_with_male (total : Nat) (men : Nat) (women : Nat) :
  total = 8 →
  men = 5 →
  women = 3 →
  Nat.choose total 2 - Nat.choose women 2 = 25 :=
by sorry

end two_person_subcommittees_with_male_l1922_192225


namespace six_x_value_l1922_192246

theorem six_x_value (x : ℝ) (h : 3 * x - 9 = 12) : 6 * x = 42 := by
  sorry

end six_x_value_l1922_192246


namespace nested_square_root_simplification_l1922_192221

theorem nested_square_root_simplification :
  Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 25 * Real.sqrt 5 := by
  sorry

end nested_square_root_simplification_l1922_192221


namespace problem_solution_l1922_192298

theorem problem_solution : ∀ A B : ℕ, 
  A = 55 * 100 + 19 * 10 → 
  B = 173 + 5 * 224 → 
  A - B = 4397 := by
  sorry

end problem_solution_l1922_192298


namespace final_savings_calculation_l1922_192203

/-- Calculate final savings after a period of time given initial savings, monthly income, and monthly expenses. -/
def calculate_final_savings (initial_savings : ℕ) (monthly_income : ℕ) (monthly_expenses : ℕ) (months : ℕ) : ℕ :=
  initial_savings + months * monthly_income - months * monthly_expenses

/-- Theorem: Given the specific financial conditions, the final savings will be 1106900 rubles. -/
theorem final_savings_calculation :
  let initial_savings : ℕ := 849400
  let monthly_income : ℕ := 45000 + 35000 + 7000 + 10000 + 13000
  let monthly_expenses : ℕ := 30000 + 10000 + 5000 + 4500 + 9000
  let months : ℕ := 5
  calculate_final_savings initial_savings monthly_income monthly_expenses months = 1106900 := by
  sorry

end final_savings_calculation_l1922_192203


namespace population_closest_to_target_in_2060_l1922_192204

def initial_population : ℕ := 500
def growth_rate : ℕ := 4
def years_per_growth : ℕ := 30
def target_population : ℕ := 10000
def initial_year : ℕ := 2000

def population_at_year (year : ℕ) : ℕ :=
  initial_population * growth_rate ^ ((year - initial_year) / years_per_growth)

theorem population_closest_to_target_in_2060 :
  ∀ year : ℕ, year ≤ 2060 → population_at_year year ≤ target_population ∧
  population_at_year 2060 > population_at_year (2060 - years_per_growth) ∧
  population_at_year (2060 + years_per_growth) > target_population :=
by sorry

end population_closest_to_target_in_2060_l1922_192204


namespace exactly_two_subsets_implies_a_values_l1922_192267

def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + a = 0}

theorem exactly_two_subsets_implies_a_values (a : ℝ) :
  (∀ S : Set ℝ, S ⊆ A a → (S = ∅ ∨ S = A a)) →
  a = -1 ∨ a = 0 ∨ a = 1 := by
  sorry

end exactly_two_subsets_implies_a_values_l1922_192267


namespace f_value_at_2012_l1922_192239

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : ∀ x : ℝ, f (x + 3) ≤ f x + 3)
variable (h2 : ∀ x : ℝ, f (x + 2) ≥ f x + 2)
variable (h3 : f 998 = 1002)

-- State the theorem
theorem f_value_at_2012 : f 2012 = 2016 := by sorry

end f_value_at_2012_l1922_192239


namespace unique_positive_solution_l1922_192244

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x := by
  sorry

end unique_positive_solution_l1922_192244


namespace cubic_function_derivative_l1922_192278

/-- Given a cubic function f(x) = ax³ + 3x² + 2, prove that if f'(-1) = 4, then a = 10/3 -/
theorem cubic_function_derivative (a : ℝ) :
  let f := λ x : ℝ => a * x^3 + 3 * x^2 + 2
  let f' := λ x : ℝ => 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by
sorry

end cubic_function_derivative_l1922_192278


namespace function_inequality_l1922_192249

theorem function_inequality (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → a + 2 * 2^x + 4^x < 0) → a < -8 := by
  sorry

end function_inequality_l1922_192249


namespace train_meeting_time_l1922_192218

/-- Represents the problem of two trains meeting on a journey from Delhi to Bombay -/
theorem train_meeting_time 
  (speed_first : ℝ) 
  (speed_second : ℝ) 
  (departure_second : ℝ) 
  (meeting_distance : ℝ) 
  (h1 : speed_first = 60) 
  (h2 : speed_second = 80) 
  (h3 : departure_second = 16.5) 
  (h4 : meeting_distance = 480) : 
  ∃ (departure_first : ℝ), 
    speed_first * (departure_second - departure_first) = meeting_distance ∧ 
    departure_first = 8.5 := by
  sorry

end train_meeting_time_l1922_192218


namespace arithmetic_sequence_problem_l1922_192233

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h1 : a 2 + a 4 = 4) 
    (h2 : a 3 + a 5 = 10) : 
  a 5 + a 7 = 22 := by
  sorry

end arithmetic_sequence_problem_l1922_192233


namespace tank_fill_time_l1922_192227

/-- Represents the time (in minutes) it takes for a pipe to fill or empty the tank -/
structure PipeRate where
  rate : ℚ
  filling : Bool

/-- Represents the state of the tank -/
structure TankState where
  filled : ℚ  -- Fraction of the tank that is filled

/-- Represents the system of pipes and the tank -/
structure PipeSystem where
  pipes : Fin 4 → PipeRate
  cycle_time : ℚ
  cycle_effect : ℚ

def apply_pipe (p : PipeRate) (t : TankState) (duration : ℚ) : TankState :=
  if p.filling then
    { filled := t.filled + duration / p.rate }
  else
    { filled := t.filled - duration / p.rate }

def apply_cycle (s : PipeSystem) (t : TankState) : TankState :=
  { filled := t.filled + s.cycle_effect }

def time_to_fill (s : PipeSystem) : ℚ :=
  s.cycle_time * (1 / s.cycle_effect)

theorem tank_fill_time (s : PipeSystem) (h1 : s.pipes 0 = ⟨20, true⟩)
    (h2 : s.pipes 1 = ⟨30, true⟩) (h3 : s.pipes 2 = ⟨15, false⟩)
    (h4 : s.pipes 3 = ⟨40, true⟩) (h5 : s.cycle_time = 16)
    (h6 : s.cycle_effect = 1/10) : time_to_fill s = 160 := by
  sorry

end tank_fill_time_l1922_192227


namespace elena_frog_count_l1922_192254

/-- Given a total number of frog eyes and the number of eyes per frog,
    calculate the number of frogs. -/
def count_frogs (total_eyes : ℕ) (eyes_per_frog : ℕ) : ℕ :=
  total_eyes / eyes_per_frog

/-- The problem statement -/
theorem elena_frog_count :
  let total_eyes : ℕ := 20
  let eyes_per_frog : ℕ := 2
  count_frogs total_eyes eyes_per_frog = 10 := by
  sorry

end elena_frog_count_l1922_192254


namespace solution_set_and_range_l1922_192279

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≥ 1 ↔ x ≤ -5/2 ∨ x ≥ 3/2) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ -1 ≤ a ∧ a ≤ 2) :=
by sorry

end solution_set_and_range_l1922_192279


namespace compute_expression_l1922_192287

theorem compute_expression : 6 * (2/3)^4 - 1/6 = 55/54 := by
  sorry

end compute_expression_l1922_192287


namespace sarahs_journey_length_l1922_192294

theorem sarahs_journey_length :
  ∀ (total : ℚ),
    (1 / 4 : ℚ) * total +   -- First part
    30 +                    -- Second part
    (1 / 6 : ℚ) * total =   -- Third part
    total →                 -- Sum of parts equals total
    total = 360 / 7 := by
  sorry

end sarahs_journey_length_l1922_192294


namespace largest_mersenne_prime_under_1000_l1922_192252

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, is_prime n ∧ p = 2^n - 1 ∧ is_prime p

theorem largest_mersenne_prime_under_1000 :
  (∀ p : ℕ, p < 1000 → is_mersenne_prime p → p ≤ 127) ∧
  is_mersenne_prime 127 :=
sorry

end largest_mersenne_prime_under_1000_l1922_192252


namespace solution_in_interval_l1922_192274

open Real

theorem solution_in_interval : ∃ (x₀ : ℝ), ∃ (k : ℤ),
  (log x₀ = 5 - 2 * x₀) ∧ 
  (x₀ > k) ∧ (x₀ < k + 1) ∧
  (k = 2) := by
  sorry

end solution_in_interval_l1922_192274


namespace die_toss_results_l1922_192211

/-- The number of faces on a fair die -/
def numFaces : ℕ := 6

/-- The number of tosses when the process stops -/
def numTosses : ℕ := 5

/-- The number of different numbers recorded when the process stops -/
def numDifferent : ℕ := 3

/-- The total number of different recording results -/
def totalResults : ℕ := 840

/-- Theorem stating the total number of different recording results -/
theorem die_toss_results :
  (numFaces = 6) →
  (numTosses = 5) →
  (numDifferent = 3) →
  (totalResults = 840) := by
  sorry

end die_toss_results_l1922_192211


namespace line_plane_parallelism_l1922_192255

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (a b : Line) (α : Plane) 
  (h1 : parallel_plane a α) 
  (h2 : parallel_line a b) 
  (h3 : ¬ contained_in b α) : 
  parallel_plane b α :=
sorry

end line_plane_parallelism_l1922_192255


namespace complex_power_sum_l1922_192232

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (Real.pi / 4)) :
  z^12 + (1 / z)^12 = -2 := by
  sorry

end complex_power_sum_l1922_192232


namespace apartment_occupancy_l1922_192236

theorem apartment_occupancy (total_floors : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ) : 
  total_floors = 12 →
  apartments_per_floor = 10 →
  total_people = 360 →
  ∃ (people_per_apartment : ℕ), 
    people_per_apartment * (apartments_per_floor * total_floors / 2 + apartments_per_floor * total_floors / 4) = total_people ∧
    people_per_apartment = 4 :=
by sorry

end apartment_occupancy_l1922_192236


namespace exam_students_count_l1922_192219

theorem exam_students_count : 
  ∀ N : ℕ,
  (N : ℝ) * 80 = 160 + (N - 8 : ℝ) * 90 →
  N = 56 :=
by
  sorry

#check exam_students_count

end exam_students_count_l1922_192219


namespace sufficient_not_necessary_l1922_192281

/-- The function f(x) = x³ + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a

/-- f is monotonically increasing on ℝ -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f a x < f a y

/-- The statement "if a > 1, then f(x) = x³ + a is monotonically increasing on ℝ" 
    is a sufficient but not necessary condition -/
theorem sufficient_not_necessary : 
  (∀ a : ℝ, a > 1 → is_monotone_increasing a) ∧ 
  (∃ a : ℝ, a ≤ 1 ∧ is_monotone_increasing a) :=
sorry

end sufficient_not_necessary_l1922_192281


namespace probability_two_white_is_three_tenths_l1922_192256

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def drawn_balls : ℕ := 2

def probability_two_white : ℚ := (white_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls)

theorem probability_two_white_is_three_tenths :
  probability_two_white = 3 / 10 := by sorry

end probability_two_white_is_three_tenths_l1922_192256


namespace sum_difference_3010_l1922_192264

/-- The sum of the first n odd counting numbers -/
def sum_odd (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The sum of the first n even counting numbers -/
def sum_even (n : ℕ) : ℕ := n * (2 * n + 2)

/-- The difference between the sum of the first n even counting numbers
    and the sum of the first n odd counting numbers -/
def sum_difference (n : ℕ) : ℕ := sum_even n - sum_odd n

theorem sum_difference_3010 :
  sum_difference 3010 = 3010 := by sorry

end sum_difference_3010_l1922_192264


namespace square_area_error_l1922_192250

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0404 := by
sorry

end square_area_error_l1922_192250


namespace length_breadth_difference_l1922_192237

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_is_24_times_breadth : area = 24 * breadth
  breadth_is_14 : breadth = 14
  area_def : area = length * breadth

/-- The difference between length and breadth is 10 meters -/
theorem length_breadth_difference (plot : RectangularPlot) : 
  plot.length - plot.breadth = 10 := by
  sorry

end length_breadth_difference_l1922_192237


namespace least_addition_for_divisibility_l1922_192214

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x < 23 ∧ (1054 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1054 + y) % 23 ≠ 0 :=
by sorry

end least_addition_for_divisibility_l1922_192214


namespace geometric_sequence_increasing_condition_l1922_192251

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_condition
  (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) :
  (is_increasing_sequence a → a 1 < a 2 ∧ a 2 < a 3) ∧
  ¬(a 1 < a 2 ∧ a 2 < a 3 → is_increasing_sequence a) :=
sorry

end geometric_sequence_increasing_condition_l1922_192251


namespace total_balloons_l1922_192216

/-- Represents the number of balloons of each color -/
structure BalloonCounts where
  gold : ℕ
  silver : ℕ
  black : ℕ
  blue : ℕ
  red : ℕ

/-- The conditions of the balloon problem -/
def balloon_problem (b : BalloonCounts) : Prop :=
  b.gold = 141 ∧
  b.silver = 2 * b.gold ∧
  b.black = 150 ∧
  b.blue = b.silver / 2 ∧
  b.red = b.blue / 3

/-- The theorem stating the total number of balloons -/
theorem total_balloons (b : BalloonCounts) 
  (h : balloon_problem b) : 
  b.gold + b.silver + b.black + b.blue + b.red = 761 := by
  sorry

#check total_balloons

end total_balloons_l1922_192216


namespace shaded_area_is_four_thirds_l1922_192291

/-- Rectangle with specific dimensions and lines forming a shaded region --/
structure ShadedRectangle where
  J : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  M : ℝ × ℝ
  h_rectangle : J.1 = 0 ∧ J.2 = 0 ∧ K.1 = 4 ∧ K.2 = 0 ∧ L.1 = 4 ∧ L.2 = 5 ∧ M.1 = 0 ∧ M.2 = 5
  h_mj : M.2 - J.2 = 2
  h_jk : K.1 - J.1 = 1
  h_kl : L.2 - K.2 = 1
  h_lm : M.1 - L.1 = 1

/-- The area of the shaded region in the rectangle --/
def shadedArea (r : ShadedRectangle) : ℝ := sorry

/-- Theorem stating that the shaded area is 4/3 --/
theorem shaded_area_is_four_thirds (r : ShadedRectangle) : shadedArea r = 4/3 := by
  sorry

end shaded_area_is_four_thirds_l1922_192291


namespace expand_expression_l1922_192212

theorem expand_expression (x : ℝ) : (x - 1) * (4 * x + 5) = 4 * x^2 + x - 5 := by
  sorry

end expand_expression_l1922_192212


namespace perpendicular_lines_from_perpendicular_planes_l1922_192268

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Plane → Plane → Prop)
variable (perpLine : Line → Plane → Prop)
variable (perpLines : Line → Line → Prop)

-- Define the property of being different planes
variable (different : Plane → Plane → Prop)

-- Define the property of being non-coincident lines
variable (nonCoincident : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (m n : Line)
  (h1 : different α β)
  (h2 : nonCoincident m n)
  (h3 : perpLine m α)
  (h4 : perpLine n β)
  (h5 : perp α β) :
  perpLines m n :=
sorry

end perpendicular_lines_from_perpendicular_planes_l1922_192268


namespace unique_number_l1922_192226

theorem unique_number : ∃! x : ℚ, x / 3 = x - 4 := by sorry

end unique_number_l1922_192226


namespace complex_multiplication_l1922_192269

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by sorry

end complex_multiplication_l1922_192269


namespace modular_inverse_15_mod_17_l1922_192288

theorem modular_inverse_15_mod_17 : ∃ x : ℕ, x ≤ 16 ∧ (15 * x) % 17 = 1 :=
by
  use 9
  sorry

end modular_inverse_15_mod_17_l1922_192288


namespace problem_statement_l1922_192222

theorem problem_statement (x y : ℝ) : 
  16 * (4 : ℝ)^x = 3^(y + 2) → y = -2 → x = -2 := by sorry

end problem_statement_l1922_192222


namespace triangle_angle_proof_l1922_192243

/-- Given a triangle with angles 45°, 3x°, and x°, prove that x = 33.75° -/
theorem triangle_angle_proof (x : ℝ) : 
  45 + 3*x + x = 180 → x = 33.75 := by
  sorry

end triangle_angle_proof_l1922_192243


namespace move_right_two_units_l1922_192285

/-- Moving a point 2 units to the right in a Cartesian coordinate system -/
theorem move_right_two_units (initial_x initial_y : ℝ) :
  let initial_point := (initial_x, initial_y)
  let final_point := (initial_x + 2, initial_y)
  initial_point = (1, 1) → final_point = (3, 1) := by
  sorry

end move_right_two_units_l1922_192285


namespace total_throw_distance_l1922_192217

/-- Proves the total distance thrown over two days is 1600 yards. -/
theorem total_throw_distance (T : ℝ) : 
  let throw_distance_T := 20
  let throw_distance_80 := 2 * throw_distance_T
  let saturday_throws := 20
  let sunday_throws := 30
  let saturday_distance := saturday_throws * throw_distance_T
  let sunday_distance := sunday_throws * throw_distance_80
  saturday_distance + sunday_distance = 1600 := by sorry

end total_throw_distance_l1922_192217


namespace sum_of_cubic_roots_l1922_192280

theorem sum_of_cubic_roots (a b c d : ℝ) (h : ∀ x : ℝ, x^3 + x^2 - 6*x - 20 = 4*x + 24 ↔ a*x^3 + b*x^2 + c*x + d = 0) :
  a ≠ 0 → (sum_of_roots : ℝ) = -b / a ∧ sum_of_roots = -1 :=
by sorry


end sum_of_cubic_roots_l1922_192280


namespace profit_maximizing_prices_l1922_192297

/-- Represents the selling price in yuan -/
def selling_price : ℝ → ℝ := id

/-- Represents the daily sales quantity as a function of selling price -/
def daily_sales (x : ℝ) : ℝ := 200 - (x - 20) * 20

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 12) * (daily_sales x)

/-- The theorem states that 19 and 23 are the only selling prices that achieve a daily profit of 1540 yuan -/
theorem profit_maximizing_prices :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  daily_profit x₁ = 1540 ∧ 
  daily_profit x₂ = 1540 ∧
  (∀ x : ℝ, daily_profit x = 1540 → (x = x₁ ∨ x = x₂)) :=
sorry

end profit_maximizing_prices_l1922_192297


namespace find_a_value_l1922_192258

theorem find_a_value (x y : ℝ) (a : ℝ) 
  (h1 : x = 2)
  (h2 : y = 1)
  (h3 : a * x - y = 3) :
  a = 2 := by
sorry

end find_a_value_l1922_192258


namespace water_fraction_after_four_replacements_l1922_192235

/-- Represents the state of the water tank -/
structure TankState where
  water : ℚ
  antifreeze : ℚ

/-- Performs one replacement operation on the tank -/
def replace (state : TankState) : TankState :=
  let removed := state.water * (5 / 20) + state.antifreeze * (5 / 20)
  { water := state.water - removed + 2.5,
    antifreeze := state.antifreeze - removed + 2.5 }

/-- The initial state of the tank -/
def initialState : TankState :=
  { water := 20, antifreeze := 0 }

/-- Performs n replacements on the tank -/
def nReplacements (n : ℕ) : TankState :=
  match n with
  | 0 => initialState
  | n + 1 => replace (nReplacements n)

theorem water_fraction_after_four_replacements :
  (nReplacements 4).water / ((nReplacements 4).water + (nReplacements 4).antifreeze) = 21 / 32 :=
by sorry

end water_fraction_after_four_replacements_l1922_192235


namespace sin_cos_sum_equals_neg_one_l1922_192271

theorem sin_cos_sum_equals_neg_one : 
  Real.sin (315 * π / 180) - Real.cos (135 * π / 180) + 2 * Real.sin (570 * π / 180) = -1 := by
  sorry

end sin_cos_sum_equals_neg_one_l1922_192271


namespace uniform_random_transformation_l1922_192253

/-- A uniform random variable on an interval -/
def UniformRandom (a b : ℝ) (X : ℝ → Prop) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → X x

theorem uniform_random_transformation (b₁ : ℝ → Prop) (b : ℝ → Prop) :
  UniformRandom 0 1 b₁ →
  (∀ x, b x ↔ ∃ y, b₁ y ∧ x = 3 * (y - 2)) →
  UniformRandom (-6) (-3) b :=
sorry

end uniform_random_transformation_l1922_192253


namespace billy_age_is_47_25_l1922_192275

-- Define Billy's and Joe's ages
def billy_age : ℝ := sorry
def joe_age : ℝ := sorry

-- State the theorem
theorem billy_age_is_47_25 :
  (billy_age = 3 * joe_age) →  -- Billy's age is three times Joe's age
  (billy_age + joe_age = 63) → -- The sum of their ages is 63 years
  (billy_age = 47.25) :=       -- Billy's age is 47.25 years
by
  sorry

end billy_age_is_47_25_l1922_192275


namespace largest_integer_solution_2x_plus_3_lt_0_l1922_192286

theorem largest_integer_solution_2x_plus_3_lt_0 :
  ∀ x : ℤ, 2 * x + 3 < 0 → x ≤ -2 :=
by sorry

end largest_integer_solution_2x_plus_3_lt_0_l1922_192286


namespace cistern_length_l1922_192296

theorem cistern_length (width depth area : ℝ) (h1 : width = 4)
    (h2 : depth = 1.25) (h3 : area = 55.5) :
  ∃ length : ℝ, length = 7 ∧ 
    area = (length * width) + 2 * (length * depth) + 2 * (width * depth) :=
by sorry

end cistern_length_l1922_192296


namespace quadratic_inequality_solution_l1922_192207

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - (2 + a) * x + 2 * a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | f a x < 0}

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) :
  (a < 2 → solution_set a = {x | a < x ∧ x < 2}) ∧
  (a = 2 → solution_set a = ∅) ∧
  (a > 2 → solution_set a = {x | 2 < x ∧ x < a}) := by
  sorry

end quadratic_inequality_solution_l1922_192207


namespace adjacent_pair_with_distinct_roots_l1922_192231

/-- Represents a 6x6 grid containing integers from 1 to 36 --/
def Grid := Fin 6 → Fin 6 → Fin 36

/-- Checks if two numbers are adjacent in a row --/
def areAdjacent (grid : Grid) (i j : Fin 6) (k : Fin 5) : Prop :=
  grid i k = j ∧ grid i (k + 1) = j + 1 ∨ grid i k = j + 1 ∧ grid i (k + 1) = j

/-- Checks if a quadratic equation has two distinct real roots --/
def hasTwoDistinctRealRoots (p q : ℕ) : Prop :=
  p * p > 4 * q

theorem adjacent_pair_with_distinct_roots (grid : Grid) :
  ∃ (i : Fin 6) (j : Fin 36) (k : Fin 5),
    areAdjacent grid j (j + 1) k ∧
    hasTwoDistinctRealRoots j (j + 1) := by
  sorry

end adjacent_pair_with_distinct_roots_l1922_192231


namespace student_guinea_pig_difference_is_126_l1922_192240

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 24

/-- The number of guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- The number of classrooms -/
def number_of_classrooms : ℕ := 6

/-- The difference between the total number of students and the total number of guinea pigs -/
def student_guinea_pig_difference : ℕ := 
  (students_per_classroom * number_of_classrooms) - (guinea_pigs_per_classroom * number_of_classrooms)

theorem student_guinea_pig_difference_is_126 : student_guinea_pig_difference = 126 := by
  sorry

end student_guinea_pig_difference_is_126_l1922_192240


namespace breakable_iff_composite_l1922_192273

def is_breakable (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ), a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ a + b = n ∧ (x : ℚ) / a + (y : ℚ) / b = 1

theorem breakable_iff_composite (n : ℕ) : is_breakable n ↔ ¬ Nat.Prime n ∧ n > 1 :=
sorry

end breakable_iff_composite_l1922_192273


namespace prime_square_sum_l1922_192202

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2,2,5), (2,5,2), (3,2,3), (3,3,2)} ∪ {(p,q,r) | p = 2 ∧ q = r ∧ q ≥ 3 ∧ Nat.Prime q}

theorem prime_square_sum (p q r : ℕ) :
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ is_perfect_square (p^q + p^r) ↔ (p,q,r) ∈ solution_set :=
sorry

end prime_square_sum_l1922_192202


namespace complex_sum_problem_l1922_192283

theorem complex_sum_problem (a b c d e f : ℂ) : 
  b = 1 →
  e = -a - 2*c →
  a + b * Complex.I + c + d * Complex.I + e + f * Complex.I = 3 + 2 * Complex.I →
  d + f = 1 := by
sorry

end complex_sum_problem_l1922_192283


namespace fixed_point_on_line_l1922_192242

/-- Proves that for any real number m, the line (2m+1)x + (m+1)y - 7m - 4 = 0 passes through the point (3, 1) -/
theorem fixed_point_on_line (m : ℝ) : (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

end fixed_point_on_line_l1922_192242


namespace quadratic_inequality_solution_l1922_192292

/-- Given a quadratic inequality ax^2 + b > 0 with solution set (-∞, -1/2) ∪ (1/3, ∞), prove ab = 24 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, a * x^2 + b > 0 ↔ x < -1/2 ∨ x > 1/3) → a * b = 24 := by
  sorry

end quadratic_inequality_solution_l1922_192292


namespace g_zeros_l1922_192261

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - 2 * x + 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + x * Real.log x

theorem g_zeros (a : ℝ) (h : a > 0) :
  (∃! x, g a x = 0 ∧ x > 0) ∧ a = Real.exp (-1) ∨
  (∀ x > 0, g a x ≠ 0) ∧ a > Real.exp (-1) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧
    ∀ x, x > 0 → g a x = 0 → x = x₁ ∨ x = x₂) ∧ 0 < a ∧ a < Real.exp (-1) :=
sorry

end g_zeros_l1922_192261


namespace triangle_with_specific_circumcircle_l1922_192257

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def circumscribed_circle_diameter (a b c : ℕ) : ℚ :=
  (a * b * c : ℚ) / ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c) : ℚ) * 4

theorem triangle_with_specific_circumcircle :
  ∀ a b c : ℕ,
    is_triangle a b c →
    circumscribed_circle_diameter a b c = 25/4 →
    (a = 5 ∧ b = 5 ∧ c = 6) ∨ (a = 5 ∧ b = 6 ∧ c = 5) ∨ (a = 6 ∧ b = 5 ∧ c = 5) :=
by sorry

end triangle_with_specific_circumcircle_l1922_192257


namespace potato_ratio_l1922_192206

theorem potato_ratio (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) 
  (h1 : total_potatoes = 24)
  (h2 : num_people = 3)
  (h3 : potatoes_per_person = 8)
  (h4 : total_potatoes = num_people * potatoes_per_person) :
  ∃ (r : ℕ), r > 0 ∧ 
    (potatoes_per_person, potatoes_per_person, potatoes_per_person) = (r, r, r) := by
  sorry

end potato_ratio_l1922_192206
