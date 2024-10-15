import Mathlib

namespace NUMINAMATH_CALUDE_herb_count_at_spring_end_l1369_136973

def spring_duration : ℕ := 6

def initial_basil : ℕ := 3
def initial_parsley : ℕ := 1
def initial_mint : ℕ := 2
def initial_rosemary : ℕ := 1
def initial_thyme : ℕ := 1

def basil_growth_rate : ℕ → ℕ := λ weeks => 2^(weeks / 2)
def parsley_growth_rate : ℕ → ℕ := λ weeks => weeks
def mint_growth_rate : ℕ → ℕ := λ weeks => 3^(weeks / 4)

def extra_basil_week : ℕ := 3
def mint_stop_week : ℕ := 3
def parsley_loss_week : ℕ := 5
def parsley_loss_amount : ℕ := 2

def final_basil_count : ℕ := initial_basil * basil_growth_rate spring_duration + 1
def final_parsley_count : ℕ := initial_parsley + parsley_growth_rate spring_duration - parsley_loss_amount
def final_mint_count : ℕ := initial_mint * mint_growth_rate mint_stop_week
def final_rosemary_count : ℕ := initial_rosemary
def final_thyme_count : ℕ := initial_thyme

theorem herb_count_at_spring_end :
  final_basil_count + final_parsley_count + final_mint_count + 
  final_rosemary_count + final_thyme_count = 35 := by
  sorry

end NUMINAMATH_CALUDE_herb_count_at_spring_end_l1369_136973


namespace NUMINAMATH_CALUDE_sweet_potato_price_is_correct_l1369_136988

/-- The price of each sweet potato in Alice's grocery order --/
def sweet_potato_price : ℚ :=
  let minimum_spend : ℚ := 35
  let chicken_price : ℚ := 6 * (3/2)
  let lettuce_price : ℚ := 3
  let tomato_price : ℚ := 5/2
  let broccoli_price : ℚ := 2 * 2
  let sprouts_price : ℚ := 5/2
  let sweet_potato_count : ℕ := 4
  let additional_spend : ℚ := 11
  let total_without_potatoes : ℚ := chicken_price + lettuce_price + tomato_price + broccoli_price + sprouts_price
  let potato_total : ℚ := minimum_spend - additional_spend - total_without_potatoes
  potato_total / sweet_potato_count

theorem sweet_potato_price_is_correct : sweet_potato_price = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_sweet_potato_price_is_correct_l1369_136988


namespace NUMINAMATH_CALUDE_quadratic_composite_zeros_l1369_136958

/-- A quadratic function f(x) = ax^2 + bx + c where a > 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The function f(x) -/
def f (q : QuadraticFunction) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- The composite function f(f(x)) -/
def f_comp_f (q : QuadraticFunction) (x : ℝ) : ℝ :=
  f q (f q x)

/-- The number of distinct real zeros of a function -/
def num_distinct_real_zeros (g : ℝ → ℝ) : ℕ := sorry

theorem quadratic_composite_zeros
  (q : QuadraticFunction)
  (h : f q (1 / q.a) < 0) :
  num_distinct_real_zeros (f_comp_f q) = 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_composite_zeros_l1369_136958


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1369_136961

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1369_136961


namespace NUMINAMATH_CALUDE_bug_travel_distance_l1369_136974

theorem bug_travel_distance (r : ℝ) (s : ℝ) (h1 : r = 65) (h2 : s = 100) :
  let d := 2 * r
  let x := Real.sqrt (d^2 - s^2)
  d + s + x = 313 :=
by sorry

end NUMINAMATH_CALUDE_bug_travel_distance_l1369_136974


namespace NUMINAMATH_CALUDE_inequality_solution_l1369_136943

theorem inequality_solution (x : ℝ) :
  x ≠ 3 →
  (x * (x + 1) / (x - 3)^2 ≥ 8 ↔ 3 < x ∧ x ≤ 24/7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1369_136943


namespace NUMINAMATH_CALUDE_ball_distribution_l1369_136993

theorem ball_distribution (a b c : ℕ) : 
  a + b + c = 45 →
  a + 2 = b - 1 ∧ a + 2 = c - 1 →
  (a, b, c) = (13, 16, 16) :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_l1369_136993


namespace NUMINAMATH_CALUDE_base_8_subtraction_example_l1369_136938

/-- Subtraction in base 8 -/
def base_8_subtraction (a b : ℕ) : ℕ :=
  sorry

/-- Conversion from base 10 to base 8 -/
def to_base_8 (n : ℕ) : ℕ :=
  sorry

/-- Conversion from base 8 to base 10 -/
def from_base_8 (n : ℕ) : ℕ :=
  sorry

theorem base_8_subtraction_example :
  base_8_subtraction (from_base_8 7463) (from_base_8 3154) = from_base_8 4317 :=
sorry

end NUMINAMATH_CALUDE_base_8_subtraction_example_l1369_136938


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1369_136986

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 - a 5 + a 15 = 20 →
  a 3 + a 19 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1369_136986


namespace NUMINAMATH_CALUDE_cost_of_roses_shoes_l1369_136971

/-- The cost of Rose's shoes given Mary and Rose's shopping details -/
theorem cost_of_roses_shoes 
  (mary_rose_total : ℝ → ℝ → Prop)  -- Mary and Rose spent the same total amount
  (mary_sunglasses_cost : ℝ)        -- Cost of each pair of Mary's sunglasses
  (mary_sunglasses_quantity : ℕ)    -- Number of pairs of sunglasses Mary bought
  (mary_jeans_cost : ℝ)             -- Cost of Mary's jeans
  (rose_cards_cost : ℝ)             -- Cost of each deck of Rose's basketball cards
  (rose_cards_quantity : ℕ)         -- Number of decks of basketball cards Rose bought
  (h1 : mary_sunglasses_cost = 50)
  (h2 : mary_sunglasses_quantity = 2)
  (h3 : mary_jeans_cost = 100)
  (h4 : rose_cards_cost = 25)
  (h5 : rose_cards_quantity = 2)
  (h6 : mary_rose_total (mary_sunglasses_cost * mary_sunglasses_quantity + mary_jeans_cost) 
                        (rose_cards_cost * rose_cards_quantity + rose_shoes_cost))
  : rose_shoes_cost = 150 := by
  sorry


end NUMINAMATH_CALUDE_cost_of_roses_shoes_l1369_136971


namespace NUMINAMATH_CALUDE_a_range_l1369_136990

theorem a_range (p : ∀ x > 0, x + 1/x ≥ a^2 - a) 
                (q : ∃ x : ℝ, x + |x - 1| = 2*a) : 
  a ∈ Set.Icc (1/2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l1369_136990


namespace NUMINAMATH_CALUDE_mcdonald_farm_weeks_l1369_136954

/-- The number of weeks required for Mcdonald's farm to produce the total number of eggs -/
def weeks_required (saly_eggs ben_eggs total_eggs : ℕ) : ℕ :=
  total_eggs / (saly_eggs + ben_eggs + ben_eggs / 2)

/-- Theorem stating that the number of weeks required is 4 -/
theorem mcdonald_farm_weeks : weeks_required 10 14 124 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mcdonald_farm_weeks_l1369_136954


namespace NUMINAMATH_CALUDE_inequality_proof_l1369_136981

theorem inequality_proof (x m : ℝ) (a b c : ℝ) :
  (∀ x, |x - 3| + |x - m| ≥ 2*m) →
  a > 0 → b > 0 → c > 0 → a + b + c = 1 →
  (∃ m_max : ℝ, m_max = 1 ∧ 
    (∀ m', (∀ x, |x - 3| + |x - m'| ≥ 2*m') → m' ≤ m_max)) ∧
  (4*a^2 + 9*b^2 + c^2 ≥ 36/49) ∧
  (4*a^2 + 9*b^2 + c^2 = 36/49 ↔ a = 9/49 ∧ b = 4/49 ∧ c = 36/49) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l1369_136981


namespace NUMINAMATH_CALUDE_problem_solution_l1369_136907

theorem problem_solution (x y : ℝ) 
  (h1 : x / 2 + 5 = 11) 
  (h2 : Real.sqrt y = x) : 
  x = 12 ∧ y = 144 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1369_136907


namespace NUMINAMATH_CALUDE_matrix_cube_proof_l1369_136982

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_cube_proof : A ^ 3 = !![(-4), 2; (-2), 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_cube_proof_l1369_136982


namespace NUMINAMATH_CALUDE_water_transfer_difference_l1369_136963

theorem water_transfer_difference (suho_original seohyun_original : ℚ) : 
  suho_original ≥ 0 →
  seohyun_original ≥ 0 →
  (suho_original - 7/3) = (seohyun_original + 7/3 + 3/2) →
  suho_original - seohyun_original = 37/6 :=
by sorry

end NUMINAMATH_CALUDE_water_transfer_difference_l1369_136963


namespace NUMINAMATH_CALUDE_tony_fish_count_l1369_136908

/-- The number of fish Tony's parents buy each year -/
def fish_bought_yearly : ℕ := 2

/-- The number of years that pass -/
def years : ℕ := 5

/-- The number of fish Tony starts with -/
def initial_fish : ℕ := 2

/-- The number of fish that die each year -/
def fish_lost_yearly : ℕ := 1

/-- The number of fish Tony has after 5 years -/
def final_fish : ℕ := 7

theorem tony_fish_count :
  initial_fish + years * (fish_bought_yearly - fish_lost_yearly) = final_fish :=
by sorry

end NUMINAMATH_CALUDE_tony_fish_count_l1369_136908


namespace NUMINAMATH_CALUDE_total_situps_is_510_l1369_136953

/-- The number of sit-ups Barney can perform in one minute -/
def barney_situps : ℕ := 45

/-- The number of minutes Barney performs sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The number of sit-ups Carrie can perform in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can perform in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ :=
  barney_situps * barney_minutes +
  carrie_situps * carrie_minutes +
  jerrie_situps * jerrie_minutes

/-- Theorem stating that the total number of sit-ups is 510 -/
theorem total_situps_is_510 : total_situps = 510 := by
  sorry

end NUMINAMATH_CALUDE_total_situps_is_510_l1369_136953


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1369_136983

theorem yellow_balls_count (total : ℕ) (red : ℕ) (yellow : ℕ) (prob : ℚ) : 
  red = 10 →
  yellow + red = total →
  prob = 2 / 5 →
  (red : ℚ) / total = prob →
  yellow = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1369_136983


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_a_range_for_positive_f_l1369_136977

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2 * |x + a|

-- Part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | -2 < x ∧ x < -2/3} := by sorry

-- Part 2
theorem a_range_for_positive_f :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 2 3, f a x > 0) → -5/2 < a ∧ a < -2 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_a_range_for_positive_f_l1369_136977


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1369_136952

theorem arithmetic_calculations : 
  (1 * (-30) - 4 * (-4) = -14) ∧ 
  ((-2)^2 - (1/7) * (-3-4) = 5) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1369_136952


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1369_136976

-- Problem 1
theorem problem_1 : 
  |Real.sqrt 3 - 2| + Real.sqrt 12 - 6 * Real.sin (30 * π / 180) + (-1/2)⁻¹ = Real.sqrt 3 - 3 :=
sorry

-- Problem 2
theorem problem_2 : 
  ∀ x : ℝ, x * (x + 6) = -5 ↔ x = -5 ∨ x = -1 :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1369_136976


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1369_136950

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Applies a horizontal and vertical shift to a parabola --/
def shift_parabola (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 2 ∧ p.h = 1 ∧ p.k = 3 →
  let p' := shift_parabola p 2 (-1)
  p'.a = 2 ∧ p'.h = -1 ∧ p'.k = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1369_136950


namespace NUMINAMATH_CALUDE_room_observation_ratio_l1369_136909

-- Define the room dimensions
def room_length : ℝ := 40
def room_width : ℝ := 40

-- Define the area observed by both guards
def area_observed_by_both : ℝ := 400

-- Define the total area of the room
def total_area : ℝ := room_length * room_width

-- Theorem to prove
theorem room_observation_ratio :
  total_area / area_observed_by_both = 4 := by
  sorry


end NUMINAMATH_CALUDE_room_observation_ratio_l1369_136909


namespace NUMINAMATH_CALUDE_gasoline_price_increase_percentage_l1369_136994

def highest_price : ℝ := 24
def lowest_price : ℝ := 12

theorem gasoline_price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_percentage_l1369_136994


namespace NUMINAMATH_CALUDE_odd_function_extension_l1369_136967

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_pos : ∀ x > 0, f x = x^3 + x + 1) :
  ∀ x < 0, f x = x^3 + x - 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1369_136967


namespace NUMINAMATH_CALUDE_bookshop_inventory_l1369_136955

theorem bookshop_inventory (books_sold : ℕ) (percentage_sold : ℚ) (initial_stock : ℕ) : 
  books_sold = 280 → percentage_sold = 2/5 → initial_stock * percentage_sold = books_sold → 
  initial_stock = 700 := by
sorry

end NUMINAMATH_CALUDE_bookshop_inventory_l1369_136955


namespace NUMINAMATH_CALUDE_sin_600_degrees_l1369_136951

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l1369_136951


namespace NUMINAMATH_CALUDE_initial_stock_calculation_l1369_136962

theorem initial_stock_calculation (sold : ℕ) (unsold_percentage : ℚ) 
  (h1 : sold = 402)
  (h2 : unsold_percentage = 665/1000) : 
  ∃ initial_stock : ℕ, 
    initial_stock = 1200 ∧ 
    (1 - unsold_percentage) * initial_stock = sold :=
by sorry

end NUMINAMATH_CALUDE_initial_stock_calculation_l1369_136962


namespace NUMINAMATH_CALUDE_seminar_attendance_l1369_136932

/-- The total number of people who attended the seminars given the attendance for math and music seminars -/
theorem seminar_attendance (math_attendees music_attendees both_attendees : ℕ) 
  (h1 : math_attendees = 75)
  (h2 : music_attendees = 61)
  (h3 : both_attendees = 12) :
  math_attendees + music_attendees - both_attendees = 124 := by
  sorry

#check seminar_attendance

end NUMINAMATH_CALUDE_seminar_attendance_l1369_136932


namespace NUMINAMATH_CALUDE_limit_equals_one_implies_a_and_b_l1369_136991

/-- Given that a and b are constants such that the limit of (ln(2-x))^2 / (x^2 + ax + b) as x approaches 1 is equal to 1, prove that a = -2 and b = 1. -/
theorem limit_equals_one_implies_a_and_b (a b : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |((Real.log (2 - x))^2) / (x^2 + a*x + b) - 1| < ε) →
  a = -2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_limit_equals_one_implies_a_and_b_l1369_136991


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1369_136979

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n, a (n + 1) = (a n : ℚ) * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℕ)
  (h_geometric : is_geometric_sequence a)
  (h_first : a 1 = 5)
  (h_fifth : a 5 = 1280) :
  a 4 = 320 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1369_136979


namespace NUMINAMATH_CALUDE_max_k_value_l1369_136920

noncomputable def f (x : ℝ) := x + x * Real.log x

theorem max_k_value (k : ℤ) :
  (∀ x > 2, k * (x - 2) < f x) → k ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l1369_136920


namespace NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l1369_136914

/-- The surface area of a sphere inscribed in a cube with edge length 2 is 8π. -/
theorem sphere_in_cube_surface_area :
  let cube_edge : ℝ := 2
  let sphere_diameter : ℝ := cube_edge
  let sphere_radius : ℝ := sphere_diameter / 2
  let sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l1369_136914


namespace NUMINAMATH_CALUDE_solve_equation_l1369_136900

/-- The original equation -/
def original_equation (x a : ℚ) : Prop :=
  (2*x - 1) / 5 + 1 = (x + a) / 2

/-- The incorrect equation due to mistake -/
def incorrect_equation (x a : ℚ) : Prop :=
  2*(2*x - 1) + 1 = 5*(x + a)

/-- Theorem stating the correct values of a and x -/
theorem solve_equation :
  ∃ (a : ℚ), (incorrect_equation (-6) a) ∧ 
  (∀ x : ℚ, original_equation x a ↔ x = 3) ∧ 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l1369_136900


namespace NUMINAMATH_CALUDE_flower_stitches_l1369_136931

/-- Proves that given the conditions, the number of stitches required to embroider one flower is 60. -/
theorem flower_stitches (
  stitches_per_minute : ℕ)
  (unicorn_stitches : ℕ)
  (godzilla_stitches : ℕ)
  (num_unicorns : ℕ)
  (num_flowers : ℕ)
  (total_minutes : ℕ)
  (h1 : stitches_per_minute = 4)
  (h2 : unicorn_stitches = 180)
  (h3 : godzilla_stitches = 800)
  (h4 : num_unicorns = 3)
  (h5 : num_flowers = 50)
  (h6 : total_minutes = 1085)
  : (total_minutes * stitches_per_minute - (num_unicorns * unicorn_stitches + godzilla_stitches)) / num_flowers = 60 :=
sorry

end NUMINAMATH_CALUDE_flower_stitches_l1369_136931


namespace NUMINAMATH_CALUDE_train_speed_l1369_136919

/-- The speed of a train crossing a platform -/
theorem train_speed (train_length platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  platform_length = 165 →
  crossing_time = 7.499400047996161 →
  ∃ (speed : ℝ), abs (speed - 132.01) < 0.01 ∧ 
  speed = (train_length + platform_length) / crossing_time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1369_136919


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l1369_136926

-- Define the line and ellipse
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 20 = 1

-- Theorem statement
theorem line_intersects_ellipse (k : ℝ) :
  ∃ x y : ℝ, line k x = y ∧ ellipse x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l1369_136926


namespace NUMINAMATH_CALUDE_truncated_prism_edges_l1369_136923

/-- Represents a truncated rectangular prism -/
structure TruncatedPrism where
  originalEdges : ℕ
  normalTruncations : ℕ
  intersectingTruncations : ℕ

/-- Calculates the number of edges after truncation -/
def edgesAfterTruncation (p : TruncatedPrism) : ℕ :=
  p.originalEdges - p.intersectingTruncations +
  p.normalTruncations * 3 + p.intersectingTruncations * 4

/-- Theorem stating that the specific truncation scenario results in 33 edges -/
theorem truncated_prism_edges :
  ∀ p : TruncatedPrism,
  p.originalEdges = 12 ∧
  p.normalTruncations = 6 ∧
  p.intersectingTruncations = 1 →
  edgesAfterTruncation p = 33 :=
by
  sorry


end NUMINAMATH_CALUDE_truncated_prism_edges_l1369_136923


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l1369_136913

def polynomial (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem real_roots_of_polynomial :
  ∀ x : ℝ, polynomial x = 0 ↔ x = 2 ∨ x = -Real.sqrt 2 ∨ x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l1369_136913


namespace NUMINAMATH_CALUDE_largest_five_digit_with_given_product_l1369_136960

/-- The product of the digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Check if a number is a five-digit integer -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_with_given_product :
  (∀ n : ℕ, is_five_digit n ∧ digit_product n = 40320 → n ≤ 98752) ∧
  is_five_digit 98752 ∧
  digit_product 98752 = 40320 := by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_given_product_l1369_136960


namespace NUMINAMATH_CALUDE_gravel_path_cost_example_l1369_136927

/-- Calculates the cost of gravelling a path inside a rectangular plot -/
def gravel_path_cost (length width path_width gravel_cost_per_sqm : ℝ) : ℝ :=
  let total_area := length * width
  let inner_area := (length - 2 * path_width) * (width - 2 * path_width)
  let path_area := total_area - inner_area
  path_area * gravel_cost_per_sqm

/-- Theorem: The cost of gravelling the path is 425 INR -/
theorem gravel_path_cost_example : 
  gravel_path_cost 110 65 2.5 0.5 = 425 := by
sorry

end NUMINAMATH_CALUDE_gravel_path_cost_example_l1369_136927


namespace NUMINAMATH_CALUDE_rosa_flowers_total_l1369_136940

theorem rosa_flowers_total (initial_flowers : Float) (additional_flowers : Float) :
  initial_flowers = 67.0 →
  additional_flowers = 90.0 →
  initial_flowers + additional_flowers = 157.0 := by
sorry

end NUMINAMATH_CALUDE_rosa_flowers_total_l1369_136940


namespace NUMINAMATH_CALUDE_boat_distance_problem_l1369_136972

/-- Proves that given a boat with speed 9 kmph in standing water, a stream with speed 1.5 kmph,
    and a round trip time of 24 hours, the distance to the destination is 105 km. -/
theorem boat_distance_problem (boat_speed : ℝ) (stream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  boat_speed = 9 →
  stream_speed = 1.5 →
  total_time = 24 →
  distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time →
  distance = 105 := by
sorry


end NUMINAMATH_CALUDE_boat_distance_problem_l1369_136972


namespace NUMINAMATH_CALUDE_B_3_2_eq_4_l1369_136924

def B : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => B m 2
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem B_3_2_eq_4 : B 3 2 = 4 := by sorry

end NUMINAMATH_CALUDE_B_3_2_eq_4_l1369_136924


namespace NUMINAMATH_CALUDE_squirrel_nut_difference_example_l1369_136970

/-- Given a tree with squirrels and nuts, calculate the difference between their quantities -/
def squirrel_nut_difference (num_squirrels num_nuts : ℕ) : ℤ :=
  (num_squirrels : ℤ) - (num_nuts : ℤ)

/-- Theorem: In a tree with 4 squirrels and 2 nuts, the difference between
    the number of squirrels and nuts is 2 -/
theorem squirrel_nut_difference_example : squirrel_nut_difference 4 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_nut_difference_example_l1369_136970


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_first_ten_l1369_136998

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_first_ten :
  arithmetic_sequence_sum (-3) 6 10 = 240 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_first_ten_l1369_136998


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1369_136969

/-- A quadratic function with vertex at (-3, 2) passing through (2, -43) has a = -9/5 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x + 3)^2 + 2) → 
  (a * 2^2 + b * 2 + c = -43) →
  a = -9/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1369_136969


namespace NUMINAMATH_CALUDE_f_composition_of_one_l1369_136980

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_composition_of_one (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x + 2) : f (f (f 1)) = 53 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_one_l1369_136980


namespace NUMINAMATH_CALUDE_equation_transformation_l1369_136992

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 - 2*x^3 - 3*x^2 + 2*x + 1 = 0 ↔ x^2 * (y^2 - y - 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l1369_136992


namespace NUMINAMATH_CALUDE_no_prime_factor_seven_mod_eight_l1369_136918

theorem no_prime_factor_seven_mod_eight (n : ℕ+) :
  ∀ p : ℕ, Prime p → p ∣ (2^(n : ℕ) + 1) → p % 8 ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_factor_seven_mod_eight_l1369_136918


namespace NUMINAMATH_CALUDE_fraction_expression_value_l1369_136942

theorem fraction_expression_value (p q : ℚ) (h : p / q = 4 / 5) :
  ∃ x : ℚ, 18 / 7 + x / (2 * q + p) = 3 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_value_l1369_136942


namespace NUMINAMATH_CALUDE_bisection_is_best_method_l1369_136901

/-- Represents a transmission line with a fault -/
structure TransmissionLine :=
  (hasElectricityAtA : Bool)
  (hasElectricityAtB : Bool)
  (hasFault : Bool)

/-- Represents different methods to locate a fault -/
inductive FaultLocationMethod
  | Method618
  | FractionMethod
  | BisectionMethod
  | BlindManClimbingMethod

/-- Determines the best method to locate a fault in a transmission line -/
def bestFaultLocationMethod (line : TransmissionLine) : FaultLocationMethod :=
  FaultLocationMethod.BisectionMethod

/-- Theorem stating that the bisection method is the best for locating a fault
    in a transmission line with electricity at A but not at B -/
theorem bisection_is_best_method (line : TransmissionLine)
  (h1 : line.hasElectricityAtA = true)
  (h2 : line.hasElectricityAtB = false)
  (h3 : line.hasFault = true) :
  bestFaultLocationMethod line = FaultLocationMethod.BisectionMethod :=
by sorry

end NUMINAMATH_CALUDE_bisection_is_best_method_l1369_136901


namespace NUMINAMATH_CALUDE_sum_of_phi_plus_one_divisors_l1369_136997

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- A divisor of n is a natural number that divides n without a remainder -/
def is_divisor (d n : ℕ) : Prop := n % d = 0

theorem sum_of_phi_plus_one_divisors (n : ℕ) :
  ∃ (divisors : Finset ℕ), 
    (∀ d ∈ divisors, is_divisor d n) ∧ 
    (Finset.card divisors = phi n + 1) ∧
    (Finset.sum divisors id = n) :=
  sorry

end NUMINAMATH_CALUDE_sum_of_phi_plus_one_divisors_l1369_136997


namespace NUMINAMATH_CALUDE_brandy_safe_caffeine_l1369_136975

/-- The maximum safe amount of caffeine that can be consumed per day (in mg) -/
def max_safe_caffeine : ℕ := 500

/-- The amount of caffeine in each energy drink (in mg) -/
def caffeine_per_drink : ℕ := 120

/-- The number of energy drinks Brandy consumed -/
def drinks_consumed : ℕ := 4

/-- The remaining amount of caffeine Brandy can safely consume (in mg) -/
def remaining_safe_caffeine : ℕ := max_safe_caffeine - (caffeine_per_drink * drinks_consumed)

theorem brandy_safe_caffeine : remaining_safe_caffeine = 20 := by
  sorry

end NUMINAMATH_CALUDE_brandy_safe_caffeine_l1369_136975


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l1369_136956

/-- Given simple interest, principal, and time, calculate the interest rate. -/
theorem calculate_interest_rate 
  (simple_interest principal time rate : ℝ) 
  (h1 : simple_interest = 400)
  (h2 : principal = 1200)
  (h3 : time = 4)
  (h4 : simple_interest = principal * rate * time / 100) :
  rate = 400 * 100 / (1200 * 4) :=
by sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l1369_136956


namespace NUMINAMATH_CALUDE_curling_survey_probability_l1369_136902

/-- Represents the survey data and selection process for the Winter Olympic Games curling interest survey. -/
structure CurlingSurvey where
  total_participants : Nat
  male_to_female_ratio : Rat
  interested_ratio : Rat
  uninterested_females : Nat
  selected_interested : Nat
  chosen_promoters : Nat

/-- Calculates the probability of selecting at least one female from the chosen promoters. -/
def probability_at_least_one_female (survey : CurlingSurvey) : Rat :=
  sorry

/-- Theorem stating that given the survey conditions, the probability of selecting at least one female is 9/14. -/
theorem curling_survey_probability (survey : CurlingSurvey) 
  (h1 : survey.total_participants = 600)
  (h2 : survey.male_to_female_ratio = 2/1)
  (h3 : survey.interested_ratio = 2/3)
  (h4 : survey.uninterested_females = 50)
  (h5 : survey.selected_interested = 8)
  (h6 : survey.chosen_promoters = 2) :
  probability_at_least_one_female survey = 9/14 :=
sorry

end NUMINAMATH_CALUDE_curling_survey_probability_l1369_136902


namespace NUMINAMATH_CALUDE_sin_over_x_satisfies_equation_l1369_136957

open Real

theorem sin_over_x_satisfies_equation (x : ℝ) (hx : x ≠ 0) :
  let y : ℝ → ℝ := fun x => sin x / x
  let y' : ℝ → ℝ := fun x => (x * cos x - sin x) / (x^2)
  x * y' x + y x = cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_over_x_satisfies_equation_l1369_136957


namespace NUMINAMATH_CALUDE_dave_ticket_difference_l1369_136903

theorem dave_ticket_difference (toys clothes : ℕ) 
  (h1 : toys = 12) 
  (h2 : clothes = 7) : 
  toys - clothes = 5 := by
  sorry

end NUMINAMATH_CALUDE_dave_ticket_difference_l1369_136903


namespace NUMINAMATH_CALUDE_min_cost_all_B_trucks_l1369_136937

-- Define the capacities of trucks A and B
def truck_A_capacity : ℝ := 5
def truck_B_capacity : ℝ := 3

-- Define the cost per ton for trucks A and B
def cost_per_ton_A : ℝ := 100
def cost_per_ton_B : ℝ := 150

-- Define the total number of trucks
def total_trucks : ℕ := 5

-- Define the cost function
def cost_function (a : ℝ) : ℝ := 50 * a + 2250

-- Theorem statement
theorem min_cost_all_B_trucks :
  ∀ a : ℝ, 0 ≤ a ∧ a ≤ total_trucks →
  cost_function 0 ≤ cost_function a :=
by sorry

end NUMINAMATH_CALUDE_min_cost_all_B_trucks_l1369_136937


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l1369_136922

theorem circle_equation_m_range (m : ℝ) :
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0 ↔ (x - 1)^2 + (y - 2)^2 = r^2) →
  m < 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l1369_136922


namespace NUMINAMATH_CALUDE_field_dimensions_l1369_136985

theorem field_dimensions (m : ℝ) : (3*m + 11) * m = 100 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_field_dimensions_l1369_136985


namespace NUMINAMATH_CALUDE_sunday_necklace_production_l1369_136928

/-- The number of necklaces made by the first machine -/
def first_machine_necklaces : ℕ := 45

/-- The ratio of necklaces made by the second machine compared to the first -/
def second_machine_ratio : ℚ := 2.4

/-- The total number of necklaces made on Sunday -/
def total_necklaces : ℕ := 153

theorem sunday_necklace_production :
  (first_machine_necklaces : ℚ) + (first_machine_necklaces : ℚ) * second_machine_ratio = (total_necklaces : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sunday_necklace_production_l1369_136928


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l1369_136904

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_factorial_eight_ten : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l1369_136904


namespace NUMINAMATH_CALUDE_area_is_72_l1369_136949

/-- A square with side length 12 and a right triangle in a plane -/
structure Configuration :=
  (square_side : ℝ)
  (square_lower_right : ℝ × ℝ)
  (triangle_base : ℝ)
  (hypotenuse_end : ℝ × ℝ)

/-- The area of the region formed by the portion of the square below the diagonal of the triangle -/
def area_below_diagonal (config : Configuration) : ℝ :=
  sorry

/-- The theorem stating the area is 72 square units -/
theorem area_is_72 (config : Configuration) 
  (h1 : config.square_side = 12)
  (h2 : config.square_lower_right = (12, 0))
  (h3 : config.triangle_base = 12)
  (h4 : config.hypotenuse_end = (24, 0)) :
  area_below_diagonal config = 72 :=
sorry

end NUMINAMATH_CALUDE_area_is_72_l1369_136949


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l1369_136948

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem largest_power_dividing_factorial :
  (∀ m : ℕ, m > 7 → ¬(divides (18^m) (factorial 30))) ∧
  (divides (18^7) (factorial 30)) := by
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l1369_136948


namespace NUMINAMATH_CALUDE_johnson_family_reunion_ratio_l1369_136934

theorem johnson_family_reunion_ratio : 
  let num_children : ℕ := 45
  let num_adults : ℕ := num_children / 3
  let adults_not_blue : ℕ := 10
  let adults_blue : ℕ := num_adults - adults_not_blue
  adults_blue / num_adults = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_reunion_ratio_l1369_136934


namespace NUMINAMATH_CALUDE_area_of_triangle_AGE_l1369_136946

/-- Square ABCD with side length 5 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (5, 0) ∧ C = (5, 5) ∧ D = (0, 5))

/-- Point E on side BC such that BE = 2 and EC = 3 -/
def E : ℝ × ℝ := (5, 2)

/-- Point G is the second intersection of circumcircle of ABE with diagonal BD -/
def G : Square → ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem area_of_triangle_AGE (s : Square) :
  triangle_area s.A (G s) E = 44.5 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AGE_l1369_136946


namespace NUMINAMATH_CALUDE_length_width_difference_l1369_136959

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

theorem length_width_difference (r : Rectangle) 
  (h1 : perimeter r = 150)
  (h2 : r.length > r.width)
  (h3 : r.width = 45)
  (h4 : r.length = 60) :
  r.length - r.width = 15 := by sorry

end NUMINAMATH_CALUDE_length_width_difference_l1369_136959


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l1369_136906

theorem logarithm_equation_solution :
  ∃ (A B C : ℕ+), 
    (Nat.gcd A.val (Nat.gcd B.val C.val) = 1) ∧
    (A.val : ℝ) * (Real.log 5 / Real.log 300) + (B.val : ℝ) * (Real.log (2 * A.val) / Real.log 300) = C.val ∧
    A.val + B.val + C.val = 4 := by
  sorry

#check logarithm_equation_solution

end NUMINAMATH_CALUDE_logarithm_equation_solution_l1369_136906


namespace NUMINAMATH_CALUDE_parabola_directrix_l1369_136929

/-- Given a parabola with equation x = (1/8)y^2, its directrix has equation x = -2 -/
theorem parabola_directrix (x y : ℝ) :
  (x = (1/8) * y^2) → (∃ (p : ℝ), p > 0 ∧ x = (1/(4*p)) * y^2 ∧ -p = -2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1369_136929


namespace NUMINAMATH_CALUDE_equation_solution_l1369_136965

theorem equation_solution (x : ℝ) (h : x ≠ -2/3) :
  (3*x + 2) / (3*x^2 - 7*x - 6) = (2*x + 1) / (3*x - 2) ↔
  x = (13 + Real.sqrt 109) / 6 ∨ x = (13 - Real.sqrt 109) / 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1369_136965


namespace NUMINAMATH_CALUDE_women_average_age_l1369_136933

theorem women_average_age (n : ℕ) (A : ℝ) (age1 age2 : ℕ) :
  n = 10 ∧ 
  age1 = 10 ∧ 
  age2 = 12 ∧ 
  (n * A - age1 - age2 + 2 * ((n * (A + 2)) - (n * A - age1 - age2))) / 2 = 21 :=
by sorry

end NUMINAMATH_CALUDE_women_average_age_l1369_136933


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1369_136941

theorem rectangular_plot_breadth : 
  ∀ (length breadth area : ℝ),
  length = 3 * breadth →
  area = length * breadth →
  area = 867 →
  breadth = 17 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1369_136941


namespace NUMINAMATH_CALUDE_family_size_l1369_136925

theorem family_size (purification_cost : ℚ) (water_per_person : ℚ) (family_cost : ℚ) :
  purification_cost = 1 →
  water_per_person = 1/2 →
  family_cost = 3 →
  (family_cost / (purification_cost * water_per_person) : ℚ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_family_size_l1369_136925


namespace NUMINAMATH_CALUDE_subtract_fractions_l1369_136987

theorem subtract_fractions : (7 : ℚ) / 9 - (5 : ℚ) / 6 = (-1 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l1369_136987


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1369_136978

theorem right_triangle_hypotenuse : ∀ (short_leg long_leg hypotenuse : ℝ),
  short_leg > 0 →
  long_leg > 0 →
  hypotenuse > 0 →
  long_leg = 2 * short_leg - 1 →
  (1 / 2) * short_leg * long_leg = 60 →
  short_leg ^ 2 + long_leg ^ 2 = hypotenuse ^ 2 →
  hypotenuse = 17 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1369_136978


namespace NUMINAMATH_CALUDE_mcnugget_theorem_l1369_136945

/-- Represents the possible package sizes for Chicken McNuggets -/
def nugget_sizes : List ℕ := [6, 9, 20]

/-- Checks if a number can be expressed as a combination of nugget sizes -/
def is_orderable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

/-- The largest number that cannot be ordered -/
def largest_unorderable : ℕ := 43

/-- Main theorem: 43 is the largest number that cannot be ordered -/
theorem mcnugget_theorem :
  (∀ m > largest_unorderable, is_orderable m) ∧
  ¬(is_orderable largest_unorderable) :=
sorry

end NUMINAMATH_CALUDE_mcnugget_theorem_l1369_136945


namespace NUMINAMATH_CALUDE_smallest_b_value_l1369_136915

theorem smallest_b_value (a b : ℝ) (h1 : 1 < a) (h2 : a < b)
  (h3 : b ≥ a + 1)
  (h4 : 1 / b + 1 / a ≤ 1) :
  b ≥ (3 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1369_136915


namespace NUMINAMATH_CALUDE_sandy_balloons_l1369_136916

/-- Given the total number of blue balloons and the number of balloons Alyssa and Sally have,
    calculate the number of balloons Sandy has. -/
theorem sandy_balloons (total : ℕ) (alyssa : ℕ) (sally : ℕ) (h1 : total = 104) (h2 : alyssa = 37) (h3 : sally = 39) :
  total - alyssa - sally = 28 := by
  sorry

end NUMINAMATH_CALUDE_sandy_balloons_l1369_136916


namespace NUMINAMATH_CALUDE_number_puzzle_l1369_136964

theorem number_puzzle (x : ℝ) : 0.5 * x = 0.25 * x + 2 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1369_136964


namespace NUMINAMATH_CALUDE_circle_center_l1369_136912

/-- The equation of a circle in the form (x - h)² + (y - k)² = r² 
    where (h, k) is the center and r is the radius -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The given equation of the circle -/
def GivenEquation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y = 16

theorem circle_center : 
  ∃ (r : ℝ), ∀ (x y : ℝ), GivenEquation x y ↔ CircleEquation 4 2 r x y :=
sorry

end NUMINAMATH_CALUDE_circle_center_l1369_136912


namespace NUMINAMATH_CALUDE_no_solution_functional_equation_l1369_136968

theorem no_solution_functional_equation :
  ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x^2 + f y) = 2*x - f y :=
by sorry

end NUMINAMATH_CALUDE_no_solution_functional_equation_l1369_136968


namespace NUMINAMATH_CALUDE_greatest_digit_sum_base_seven_l1369_136905

def base_seven_representation (n : ℕ) : List ℕ := sorry

def digit_sum (digits : List ℕ) : ℕ := sorry

def is_valid_base_seven (digits : List ℕ) : Prop := sorry

theorem greatest_digit_sum_base_seven :
  ∃ (n : ℕ), n < 2890 ∧
    (∀ (m : ℕ), m < 2890 →
      digit_sum (base_seven_representation m) ≤ digit_sum (base_seven_representation n)) ∧
    digit_sum (base_seven_representation n) = 23 :=
  sorry

end NUMINAMATH_CALUDE_greatest_digit_sum_base_seven_l1369_136905


namespace NUMINAMATH_CALUDE_optimal_arrangement_l1369_136939

/-- Represents the housekeeping service company scenario -/
structure CleaningCompany where
  total_cleaners : ℕ
  large_rooms_per_cleaner : ℕ
  small_rooms_per_cleaner : ℕ
  large_room_payment : ℕ
  small_room_payment : ℕ

/-- Calculates the daily income based on the number of cleaners assigned to large rooms -/
def daily_income (company : CleaningCompany) (x : ℕ) : ℕ :=
  company.large_room_payment * company.large_rooms_per_cleaner * x +
  company.small_room_payment * company.small_rooms_per_cleaner * (company.total_cleaners - x)

/-- The main theorem to prove -/
theorem optimal_arrangement (company : CleaningCompany) (x : ℕ) :
  company.total_cleaners = 16 ∧
  company.large_rooms_per_cleaner = 4 ∧
  company.small_rooms_per_cleaner = 5 ∧
  company.large_room_payment = 80 ∧
  company.small_room_payment = 60 ∧
  x = 10 →
  daily_income company x = 5000 := by sorry

end NUMINAMATH_CALUDE_optimal_arrangement_l1369_136939


namespace NUMINAMATH_CALUDE_man_age_year_l1369_136921

theorem man_age_year (x : ℕ) (birth_year : ℕ) : 
  (1850 ≤ birth_year) ∧ (birth_year ≤ 1900) →
  (x^2 = birth_year + x) →
  (birth_year + x = 1892) := by
  sorry

end NUMINAMATH_CALUDE_man_age_year_l1369_136921


namespace NUMINAMATH_CALUDE_reading_time_per_day_l1369_136910

-- Define the given conditions
def num_books : ℕ := 3
def num_days : ℕ := 10
def reading_rate : ℕ := 100 -- words per hour
def book1_words : ℕ := 200
def book2_words : ℕ := 400
def book3_words : ℕ := 300

-- Define the theorem
theorem reading_time_per_day :
  let total_words := book1_words + book2_words + book3_words
  let total_hours := total_words / reading_rate
  let total_minutes := total_hours * 60
  total_minutes / num_days = 54 := by
sorry


end NUMINAMATH_CALUDE_reading_time_per_day_l1369_136910


namespace NUMINAMATH_CALUDE_investment_profit_ratio_l1369_136989

/-- Represents a partner's investment details -/
structure Partner where
  investment : ℚ
  time : ℕ

/-- Calculates the profit ratio of two partners -/
def profitRatio (p q : Partner) : ℚ × ℚ :=
  let pProfit := p.investment * p.time
  let qProfit := q.investment * q.time
  (pProfit, qProfit)

theorem investment_profit_ratio :
  let p : Partner := ⟨7, 5⟩
  let q : Partner := ⟨5, 14⟩
  profitRatio p q = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_investment_profit_ratio_l1369_136989


namespace NUMINAMATH_CALUDE_smallest_positive_integer_e_l1369_136999

theorem smallest_positive_integer_e (a b c d e : ℤ) : 
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = -3 ∨ x = 7 ∨ x = 11 ∨ x = -1/4) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 ∧ 
    (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e' = 0 ↔ x = -3 ∨ x = 7 ∨ x = 11 ∨ x = -1/4) →
    e' ≥ e) →
  e = 231 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_e_l1369_136999


namespace NUMINAMATH_CALUDE_circle_equation_l1369_136911

/-- The equation of a circle with center (2, -1) and tangent to the line x - y + 1 = 0 is (x-2)² + (y+1)² = 8 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -1)
  let line (x y : ℝ) := x - y + 1 = 0
  let is_tangent (c : ℝ × ℝ) (r : ℝ) (l : ℝ → ℝ → Prop) := 
    ∃ p : ℝ × ℝ, l p.1 p.2 ∧ (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2
  let circle_eq (c : ℝ × ℝ) (r : ℝ) (x y : ℝ) := 
    (x - c.1)^2 + (y - c.2)^2 = r^2
  ∃ r : ℝ, is_tangent center r line → 
    circle_eq center r x y ↔ (x - 2)^2 + (y + 1)^2 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1369_136911


namespace NUMINAMATH_CALUDE_average_speed_of_trip_l1369_136944

/-- Proves that the average speed of a trip is 16 km/h given the specified conditions -/
theorem average_speed_of_trip (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) 
    (second_part_speed : ℝ) (h1 : total_distance = 400) (h2 : first_part_distance = 100) 
    (h3 : first_part_speed = 20) (h4 : second_part_speed = 15) : 
    total_distance / (first_part_distance / first_part_speed + 
    (total_distance - first_part_distance) / second_part_speed) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_of_trip_l1369_136944


namespace NUMINAMATH_CALUDE_point_distance_l1369_136966

-- Define the points as real numbers representing their positions on a line
variable (A B C D : ℝ)

-- Define the conditions
variable (h_order : A < B ∧ B < C ∧ C < D)
variable (h_ratio : (B - A) / (C - B) = (D - A) / (D - C))
variable (h_AC : C - A = 3)
variable (h_BD : D - B = 4)

-- State the theorem
theorem point_distance (A B C D : ℝ) 
  (h_order : A < B ∧ B < C ∧ C < D)
  (h_ratio : (B - A) / (C - B) = (D - A) / (D - C))
  (h_AC : C - A = 3)
  (h_BD : D - B = 4) : 
  D - A = 6 := by sorry

end NUMINAMATH_CALUDE_point_distance_l1369_136966


namespace NUMINAMATH_CALUDE_mechanic_work_hours_l1369_136936

theorem mechanic_work_hours (rate1 rate2 total_hours total_charge : ℕ) 
  (h1 : rate1 = 45)
  (h2 : rate2 = 85)
  (h3 : total_hours = 20)
  (h4 : total_charge = 1100) :
  ∃ (hours1 hours2 : ℕ), 
    hours1 + hours2 = total_hours ∧ 
    rate1 * hours1 + rate2 * hours2 = total_charge ∧
    hours2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_work_hours_l1369_136936


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1369_136947

theorem negation_of_proposition (p : Prop) : 
  (¬(∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0)) ↔ 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1369_136947


namespace NUMINAMATH_CALUDE_sqrt_5_greater_than_2_l1369_136996

theorem sqrt_5_greater_than_2 : Real.sqrt 5 > 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_greater_than_2_l1369_136996


namespace NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l1369_136995

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem statement
theorem root_in_interval_implies_m_range :
  ∀ m : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ f x + m = 0) → -2 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l1369_136995


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l1369_136917

/-- The line equation 2y - 3x = 15 intersects the x-axis at the point (-5, 0) -/
theorem line_intersects_x_axis :
  ∃ (x : ℝ), 2 * 0 - 3 * x = 15 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l1369_136917


namespace NUMINAMATH_CALUDE_inequality_proof_l1369_136935

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1369_136935


namespace NUMINAMATH_CALUDE_weeks_per_month_l1369_136930

theorem weeks_per_month (months : ℕ) (weekly_rate : ℚ) (monthly_rate : ℚ) (savings : ℚ) :
  months = 3 ∧ 
  weekly_rate = 280 ∧ 
  monthly_rate = 1000 ∧
  savings = 360 →
  (months * monthly_rate + savings) / (months * weekly_rate) = 4 := by
sorry

end NUMINAMATH_CALUDE_weeks_per_month_l1369_136930


namespace NUMINAMATH_CALUDE_nicki_running_mileage_nicki_second_half_mileage_l1369_136984

/-- Calculates the weekly mileage for the second half of the year given the conditions -/
theorem nicki_running_mileage (total_weeks : ℕ) (first_half_weeks : ℕ) 
  (first_half_weekly_miles : ℕ) (total_annual_miles : ℕ) : ℕ :=
  let second_half_weeks := total_weeks - first_half_weeks
  let first_half_total_miles := first_half_weekly_miles * first_half_weeks
  let second_half_total_miles := total_annual_miles - first_half_total_miles
  second_half_total_miles / second_half_weeks

/-- Proves that Nicki ran 30 miles per week in the second half of the year -/
theorem nicki_second_half_mileage :
  nicki_running_mileage 52 26 20 1300 = 30 := by
  sorry

end NUMINAMATH_CALUDE_nicki_running_mileage_nicki_second_half_mileage_l1369_136984
