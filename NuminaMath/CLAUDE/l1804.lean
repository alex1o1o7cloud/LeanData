import Mathlib

namespace mystery_number_problem_l1804_180410

theorem mystery_number_problem (x : ℝ) : (x + 12) / 8 = 9 → 35 - x / 2 = 5 := by
  sorry

end mystery_number_problem_l1804_180410


namespace function_and_inequality_l1804_180488

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- Define the solution set condition
def solution_set (m : ℝ) : Set ℝ := {x | f m (x + 2) ≥ 0}

-- State the theorem
theorem function_and_inequality (m a b c : ℝ) : 
  (solution_set m = Set.Icc (-1) 1) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b + c = m) →
  (m = 1 ∧ a + 2*b + 3*c ≥ 9) := by
  sorry


end function_and_inequality_l1804_180488


namespace remaining_movie_time_l1804_180455

def movie_length : ℕ := 120
def session1 : ℕ := 35
def session2 : ℕ := 20
def session3 : ℕ := 15

theorem remaining_movie_time :
  movie_length - (session1 + session2 + session3) = 50 := by
  sorry

end remaining_movie_time_l1804_180455


namespace cyclic_iff_perpendicular_l1804_180428

-- Define the basic structures
structure Point := (x : ℝ) (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties
def is_convex (q : Quadrilateral) : Prop := sorry

def are_perpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

def is_intersection (p : Point) (p1 p2 p3 p4 : Point) : Prop := sorry

def is_midpoint (m : Point) (p1 p2 : Point) : Prop := sorry

def is_cyclic (q : Quadrilateral) : Prop := sorry

-- Main theorem
theorem cyclic_iff_perpendicular (q : Quadrilateral) (P M : Point) :
  is_convex q →
  are_perpendicular q.A q.C q.B q.D →
  is_intersection P q.A q.C q.B q.D →
  is_midpoint M q.A q.B →
  (is_cyclic q ↔ are_perpendicular P M q.D q.C) :=
by sorry

end cyclic_iff_perpendicular_l1804_180428


namespace water_remaining_after_required_pourings_l1804_180402

/-- Represents the fraction of water remaining after n pourings -/
def waterRemaining (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- The number of pourings required to reach exactly one-fifth of the original water -/
def requiredPourings : ℕ := 8

theorem water_remaining_after_required_pourings :
  waterRemaining requiredPourings = 1 / 5 := by
  sorry


end water_remaining_after_required_pourings_l1804_180402


namespace overlap_area_is_half_unit_l1804_180424

-- Define the grid and triangles
def Grid := Fin 3 × Fin 3

def Triangle1 : Set Grid := {(0, 2), (2, 0), (0, 0)}
def Triangle2 : Set Grid := {(2, 2), (0, 0), (1, 0)}

-- Define the area of overlap
def overlap_area (t1 t2 : Set Grid) : ℝ :=
  sorry

-- Theorem statement
theorem overlap_area_is_half_unit :
  overlap_area Triangle1 Triangle2 = 1/2 := by
  sorry

end overlap_area_is_half_unit_l1804_180424


namespace apple_count_l1804_180463

theorem apple_count (red_apples green_apples total_apples : ℕ) : 
  red_apples = 16 →
  green_apples = red_apples + 12 →
  total_apples = red_apples + green_apples →
  total_apples = 44 := by
  sorry

end apple_count_l1804_180463


namespace arithmetic_calculations_l1804_180448

theorem arithmetic_calculations :
  (23 + (-13) + (-17) + 8 = 1) ∧
  (-2^3 - (1 + 0.5) / (1/3) * (-3) = 11/2) := by sorry

end arithmetic_calculations_l1804_180448


namespace fourteenth_root_unity_l1804_180408

theorem fourteenth_root_unity (n : ℕ) : 
  0 ≤ n ∧ n ≤ 13 → 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * π / 14)) → 
  n = 4 := by sorry

end fourteenth_root_unity_l1804_180408


namespace lucky_larry_problem_l1804_180442

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 →
  (a - (b - (c - (d + e))) = a - b - c - d + e) →
  e = 3 := by
sorry

end lucky_larry_problem_l1804_180442


namespace sqrt_three_irrational_l1804_180468

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by sorry

end sqrt_three_irrational_l1804_180468


namespace complex_function_property_l1804_180426

/-- A function g on complex numbers defined by g(z) = (c+di)z, where c and d are real numbers. -/
def g (c d : ℝ) (z : ℂ) : ℂ := (c + d * Complex.I) * z

/-- The theorem stating that if g(z) = (c+di)z where c and d are real numbers, 
    and for all complex z, |g(z) - z| = |g(z)|, and |c+di| = 7, then d^2 = 195/4. -/
theorem complex_function_property (c d : ℝ) : 
  (∀ z : ℂ, Complex.abs (g c d z - z) = Complex.abs (g c d z)) → 
  Complex.abs (c + d * Complex.I) = 7 → 
  d^2 = 195/4 := by
  sorry

end complex_function_property_l1804_180426


namespace jack_hand_in_amount_l1804_180427

/-- Represents the number of bills of each denomination in the till -/
structure TillContents where
  hundreds : Nat
  fifties : Nat
  twenties : Nat
  tens : Nat
  fives : Nat
  ones : Nat

/-- Calculates the total value of the bills in the till -/
def totalValue (t : TillContents) : Nat :=
  100 * t.hundreds + 50 * t.fifties + 20 * t.twenties + 10 * t.tens + 5 * t.fives + t.ones

/-- Calculates the amount to be handed in to the main office -/
def amountToHandIn (t : TillContents) (amountToLeave : Nat) : Nat :=
  totalValue t - amountToLeave

/-- Theorem stating that given Jack's till contents and the amount to leave,
    the amount to hand in is $142 -/
theorem jack_hand_in_amount :
  let jacksTill : TillContents := {
    hundreds := 2,
    fifties := 1,
    twenties := 5,
    tens := 3,
    fives := 7,
    ones := 27
  }
  let amountToLeave := 300
  amountToHandIn jacksTill amountToLeave = 142 := by
  sorry


end jack_hand_in_amount_l1804_180427


namespace missing_shirts_is_eight_l1804_180473

/-- Represents the laundry problem with given conditions -/
structure LaundryProblem where
  trousers_count : ℕ
  total_bill : ℕ
  shirt_cost : ℕ
  trouser_cost : ℕ
  claimed_shirts : ℕ

/-- Calculates the number of missing shirts -/
def missing_shirts (p : LaundryProblem) : ℕ :=
  let total_trouser_cost := p.trousers_count * p.trouser_cost
  let total_shirt_cost := p.total_bill - total_trouser_cost
  let actual_shirts := total_shirt_cost / p.shirt_cost
  actual_shirts - p.claimed_shirts

/-- Theorem stating that the number of missing shirts is 8 -/
theorem missing_shirts_is_eight :
  ∃ (p : LaundryProblem),
    p.trousers_count = 10 ∧
    p.total_bill = 140 ∧
    p.shirt_cost = 5 ∧
    p.trouser_cost = 9 ∧
    p.claimed_shirts = 2 ∧
    missing_shirts p = 8 := by
  sorry

end missing_shirts_is_eight_l1804_180473


namespace max_min_difference_l1804_180416

theorem max_min_difference (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 
  let f := fun (x y z : ℝ) => x*y + y*z + z*x
  ∃ (M m : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → f x y z ≤ M) ∧
               (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → m ≤ f x y z) ∧
               M - m = 3/2 :=
by sorry

end max_min_difference_l1804_180416


namespace sum_of_fractions_equals_three_eighths_l1804_180409

theorem sum_of_fractions_equals_three_eighths :
  let sum := (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) +
             (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ))
  sum = 3 / 8 := by
  sorry

end sum_of_fractions_equals_three_eighths_l1804_180409


namespace three_digit_permutation_sum_divisible_by_37_l1804_180487

theorem three_digit_permutation_sum_divisible_by_37 (a b c : ℕ) 
  (h1 : 0 < a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9) :
  37 ∣ (100*a + 10*b + c) + 
       (100*a + 10*c + b) + 
       (100*b + 10*a + c) + 
       (100*b + 10*c + a) + 
       (100*c + 10*a + b) + 
       (100*c + 10*b + a) :=
by sorry

end three_digit_permutation_sum_divisible_by_37_l1804_180487


namespace circle_square_tangency_l1804_180498

theorem circle_square_tangency (r : ℝ) (s : ℝ) 
  (hr : r = 13) (hs : s = 18) : 
  let d := Real.sqrt (r^2 - (s - r)^2)
  (s - d = 1) ∧ d = 17 := by sorry

end circle_square_tangency_l1804_180498


namespace radical_product_equals_27_l1804_180400

theorem radical_product_equals_27 : Real.sqrt (Real.sqrt (Real.sqrt 27 * 27) * 81) * Real.sqrt 9 = 27 := by
  sorry

end radical_product_equals_27_l1804_180400


namespace cube_root_of_product_rewrite_cube_root_l1804_180474

theorem cube_root_of_product (a b c : ℕ) : 
  (a^9 * b^3 * c^3 : ℝ)^(1/3) = (a^3 * b * c : ℝ) :=
by sorry

theorem rewrite_cube_root : (2^9 * 5^3 * 7^3 : ℝ)^(1/3) = 280 :=
by sorry

end cube_root_of_product_rewrite_cube_root_l1804_180474


namespace pet_ownership_proof_l1804_180489

/-- The number of people owning only cats and dogs -/
def cats_and_dogs_owners : ℕ := 5

theorem pet_ownership_proof (total_owners : ℕ) (only_dogs : ℕ) (only_cats : ℕ) 
  (cats_dogs_snakes : ℕ) (total_snakes : ℕ) 
  (h1 : total_owners = 59)
  (h2 : only_dogs = 15)
  (h3 : only_cats = 10)
  (h4 : cats_dogs_snakes = 3)
  (h5 : total_snakes = 29) :
  cats_and_dogs_owners = total_owners - (only_dogs + only_cats + cats_dogs_snakes + (total_snakes - cats_dogs_snakes)) :=
by sorry

end pet_ownership_proof_l1804_180489


namespace dans_initial_marbles_l1804_180452

/-- The number of marbles Dan gave to Mary -/
def marbles_given : ℕ := 14

/-- The number of marbles Dan has now -/
def marbles_remaining : ℕ := 50

/-- The initial number of marbles Dan had -/
def initial_marbles : ℕ := marbles_given + marbles_remaining

theorem dans_initial_marbles : initial_marbles = 64 := by
  sorry

end dans_initial_marbles_l1804_180452


namespace rectangular_solid_edge_sum_l1804_180411

/-- A rectangular solid with volume 512 cm³, surface area 384 cm², and dimensions in geometric progression has a sum of edge lengths equal to 96 cm. -/
theorem rectangular_solid_edge_sum : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    a * b * c = 512 →
    2 * (a * b + b * c + a * c) = 384 →
    ∃ (r : ℝ), r > 0 ∧ b = a * r ∧ c = b * r →
    4 * (a + b + c) = 96 :=
by sorry


end rectangular_solid_edge_sum_l1804_180411


namespace box_volume_condition_unique_x_existence_l1804_180406

theorem box_volume_condition (x : ℕ) : Bool := 
  (x > 5) ∧ ((x + 5) * (x - 5) * (x^2 + 25) < 700)

theorem unique_x_existence : 
  ∃! x : ℕ, box_volume_condition x := by
  sorry

end box_volume_condition_unique_x_existence_l1804_180406


namespace workshop_workers_l1804_180429

/-- The total number of workers in a workshop with specific salary conditions -/
theorem workshop_workers (total_avg : ℕ) (tech_count : ℕ) (tech_avg : ℕ) (non_tech_avg : ℕ) :
  total_avg = 8000 →
  tech_count = 7 →
  tech_avg = 18000 →
  non_tech_avg = 6000 →
  ∃ (total_workers : ℕ), total_workers = 42 ∧
    total_workers * total_avg = tech_count * tech_avg + (total_workers - tech_count) * non_tech_avg :=
by sorry

end workshop_workers_l1804_180429


namespace ryan_analysis_time_l1804_180449

/-- The number of individuals Ryan is analyzing -/
def num_individuals : ℕ := 3

/-- The number of bones in each individual -/
def bones_per_individual : ℕ := 206

/-- The time (in hours) Ryan spends on initial analysis per bone -/
def initial_analysis_time : ℚ := 1

/-- The additional time (in hours) Ryan spends on research per bone -/
def additional_research_time : ℚ := 1/2

/-- The total time Ryan needs for his analysis -/
def total_analysis_time : ℚ :=
  (num_individuals * bones_per_individual) * (initial_analysis_time + additional_research_time)

theorem ryan_analysis_time : total_analysis_time = 927 := by
  sorry

end ryan_analysis_time_l1804_180449


namespace quadratic_root_form_l1804_180443

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 + 6*x + d = 0 ↔ x = (-6 + Real.sqrt d) / 2 ∨ x = (-6 - Real.sqrt d) / 2) →
  d = 36/5 := by
sorry

end quadratic_root_form_l1804_180443


namespace product_of_six_integers_square_sum_l1804_180421

theorem product_of_six_integers_square_sum (ints : Finset ℕ) : 
  ints = {1, 2, 3, 4, 5, 6, 7, 8, 9} →
  ∃ (A B : Finset ℕ), 
    A ⊆ ints ∧ B ⊆ ints ∧
    A.card = 6 ∧ B.card = 6 ∧
    A ≠ B ∧
    (∃ p : ℕ, (A.prod id : ℕ) = p^2) ∧
    (∃ q : ℕ, (B.prod id : ℕ) = q^2) ∧
    ∃ (p q : ℕ), 
      (A.prod id : ℕ) = p^2 ∧
      (B.prod id : ℕ) = q^2 ∧
      p + q = 108 :=
by sorry

end product_of_six_integers_square_sum_l1804_180421


namespace cryptarithm_solution_l1804_180430

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The cryptarithm equations -/
def cryptarithm (A B C D E F G H J : Digit) : Prop :=
  (A.val * 10 + B.val) * (C.val * 10 + A.val) = D.val * 100 + E.val * 10 + B.val ∧
  F.val * 10 + C.val - (D.val * 10 + G.val) = D.val ∧
  E.val * 10 + G.val + H.val * 10 + J.val = A.val * 100 + A.val * 10 + G.val

/-- All digits are different -/
def all_different (A B C D E F G H J : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ J ∧
  H ≠ J

theorem cryptarithm_solution :
  ∃! (A B C D E F G H J : Digit),
    cryptarithm A B C D E F G H J ∧
    all_different A B C D E F G H J ∧
    A.val = 1 ∧ B.val = 7 ∧ C.val = 2 ∧ D.val = 3 ∧
    E.val = 5 ∧ F.val = 4 ∧ G.val = 9 ∧ H.val = 6 ∧ J.val = 0 :=
by sorry

end cryptarithm_solution_l1804_180430


namespace circle_line_intersection_l1804_180469

/-- The equation of circle C is x^2 + y^2 + 8x + 15 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 8*x + 15 = 0

/-- The equation of the line is y = kx - 2 -/
def line (k x y : ℝ) : Prop := y = k*x - 2

/-- A point (x, y) is on the line y = kx - 2 -/
def point_on_line (k x y : ℝ) : Prop := line k x y

/-- The distance between two points (x1, y1) and (x2, y2) is less than or equal to r -/
def distance_le (x1 y1 x2 y2 r : ℝ) : Prop :=
  (x1 - x2)^2 + (y1 - y2)^2 ≤ r^2

theorem circle_line_intersection (k : ℝ) : 
  (∃ x y : ℝ, point_on_line k x y ∧ 
    (∃ x0 y0 : ℝ, circle_C x0 y0 ∧ distance_le x y x0 y0 1)) →
  -4/3 ≤ k ∧ k ≤ 0 := by sorry

end circle_line_intersection_l1804_180469


namespace red_light_probability_is_two_fifths_l1804_180405

/-- The duration of the red light in seconds -/
def red_duration : ℕ := 30

/-- The duration of the yellow light in seconds -/
def yellow_duration : ℕ := 5

/-- The duration of the green light in seconds -/
def green_duration : ℕ := 40

/-- The total duration of one traffic light cycle -/
def total_duration : ℕ := red_duration + yellow_duration + green_duration

/-- The probability of seeing a red light -/
def red_light_probability : ℚ := red_duration / total_duration

theorem red_light_probability_is_two_fifths :
  red_light_probability = 2 / 5 := by sorry

end red_light_probability_is_two_fifths_l1804_180405


namespace triangle_problem_l1804_180420

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that
    under certain conditions, angle A is π/4 and the area is 9/4. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  a = 3 →
  b^2 + c^2 - a^2 - Real.sqrt 2 * b * c = 0 →
  Real.sin B^2 + Real.sin C^2 = 2 * Real.sin A^2 →
  A = π / 4 ∧ 
  (1/2) * b * c * Real.sin A = 9/4 :=
by sorry

end triangle_problem_l1804_180420


namespace max_three_digit_quotient_l1804_180480

theorem max_three_digit_quotient :
  ∃ (a b c : ℕ), 
    a > 5 ∧ b > 5 ∧ c > 5 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∀ (x y z : ℕ), 
      x > 5 ∧ y > 5 ∧ z > 5 ∧ 
      x ≠ y ∧ y ≠ z ∧ x ≠ z →
      (100 * a + 10 * b + c : ℚ) / (a + b + c) ≥ (100 * x + 10 * y + z : ℚ) / (x + y + z) ∧
    (100 * a + 10 * b + c : ℚ) / (a + b + c) = 41.125 := by
  sorry

end max_three_digit_quotient_l1804_180480


namespace opposite_numbers_system_solution_l1804_180460

theorem opposite_numbers_system_solution :
  ∀ (x y k : ℝ),
  (x = -y) →
  (2 * x + 5 * y = k) →
  (x - 3 * y = 16) →
  (k = -12) :=
by sorry

end opposite_numbers_system_solution_l1804_180460


namespace decorations_count_l1804_180486

/-- The number of pieces of tinsel in each box -/
def tinsel_per_box : ℕ := 4

/-- The number of Christmas trees in each box -/
def trees_per_box : ℕ := 1

/-- The number of snow globes in each box -/
def globes_per_box : ℕ := 5

/-- The number of families receiving a box -/
def families : ℕ := 11

/-- The number of boxes given to the community center -/
def community_boxes : ℕ := 1

/-- The total number of decorations handed out -/
def total_decorations : ℕ := (tinsel_per_box + trees_per_box + globes_per_box) * (families + community_boxes)

theorem decorations_count : total_decorations = 120 := by
  sorry

end decorations_count_l1804_180486


namespace binomial_30_3_l1804_180454

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l1804_180454


namespace regular_polygon_with_20_diagonals_has_8_sides_l1804_180414

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 20 diagonals has 8 sides -/
theorem regular_polygon_with_20_diagonals_has_8_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 20 → n = 8 := by
  sorry

end regular_polygon_with_20_diagonals_has_8_sides_l1804_180414


namespace music_tool_cost_l1804_180415

/-- The cost of Joan's music tool purchase -/
theorem music_tool_cost (trumpet_cost song_book_cost total_spent : ℚ)
  (h1 : trumpet_cost = 149.16)
  (h2 : song_book_cost = 4.14)
  (h3 : total_spent = 163.28) :
  total_spent - (trumpet_cost + song_book_cost) = 9.98 := by
  sorry

end music_tool_cost_l1804_180415


namespace wholesale_price_is_correct_l1804_180465

/-- The wholesale price of a pen -/
def wholesale_price : ℝ := 2.5

/-- The retail price of one pen -/
def retail_price_one : ℝ := 5

/-- The retail price of three pens -/
def retail_price_three : ℝ := 10

/-- Theorem stating that the wholesale price of a pen is 2.5 rubles -/
theorem wholesale_price_is_correct : 
  (retail_price_one - wholesale_price = retail_price_three - 3 * wholesale_price) ∧
  wholesale_price = 2.5 := by
  sorry

end wholesale_price_is_correct_l1804_180465


namespace cube_with_tunnel_surface_area_l1804_180435

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a tunnel drilled through it -/
structure CubeWithTunnel where
  sideLength : ℝ
  tunnelVertices : Fin 3 → Point3D

/-- Calculates the surface area of a cube with a tunnel drilled through it -/
def surfaceArea (cube : CubeWithTunnel) : ℝ := sorry

/-- The main theorem stating that the surface area of the cube with tunnel is 864 -/
theorem cube_with_tunnel_surface_area :
  ∃ (cube : CubeWithTunnel),
    cube.sideLength = 12 ∧
    (cube.tunnelVertices 0).x = 3 ∧ (cube.tunnelVertices 0).y = 0 ∧ (cube.tunnelVertices 0).z = 0 ∧
    (cube.tunnelVertices 1).x = 0 ∧ (cube.tunnelVertices 1).y = 12 ∧ (cube.tunnelVertices 1).z = 0 ∧
    (cube.tunnelVertices 2).x = 0 ∧ (cube.tunnelVertices 2).y = 0 ∧ (cube.tunnelVertices 2).z = 3 ∧
    surfaceArea cube = 864 := by
  sorry

end cube_with_tunnel_surface_area_l1804_180435


namespace xyz_sum_sqrt_l1804_180459

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 15)
  (h2 : z + x = 16)
  (h3 : x + y = 17) :
  Real.sqrt (x * y * z * (x + y + z)) = 24 * Real.sqrt 42 := by
  sorry

end xyz_sum_sqrt_l1804_180459


namespace quadratic_inequality_l1804_180470

/-- A quadratic function with axis of symmetry at x = 1 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The axis of symmetry of f is at x = 1 -/
axiom axis_of_symmetry (b c : ℝ) : ∀ x, f b c (1 + x) = f b c (1 - x)

/-- The inequality f(1) < f(2) < f(-1) holds for the quadratic function f -/
theorem quadratic_inequality (b c : ℝ) : f b c 1 < f b c 2 ∧ f b c 2 < f b c (-1) := by
  sorry

end quadratic_inequality_l1804_180470


namespace hyperbola_focal_length_l1804_180438

theorem hyperbola_focal_length : 
  let a : ℝ := Real.sqrt 10
  let b : ℝ := Real.sqrt 2
  let c : ℝ := Real.sqrt (a^2 + b^2)
  2 * c = 4 * Real.sqrt 3 := by sorry

end hyperbola_focal_length_l1804_180438


namespace boat_speed_in_still_water_l1804_180485

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 4 →
  downstream_distance = 9.6 →
  downstream_time = 24 / 60 →
  ∃ (boat_speed : ℝ), boat_speed = 20 ∧ (boat_speed + current_speed) * downstream_time = downstream_distance :=
by sorry

end boat_speed_in_still_water_l1804_180485


namespace number_division_property_l1804_180467

theorem number_division_property : ∃ (n : ℕ), 
  let sum := 2468 + 1375
  let diff := 2468 - 1375
  n = 12609027 ∧
  n / sum = 3 * diff ∧
  n % sum = 150 ∧
  (n - 150) / sum = 5 * diff :=
by sorry

end number_division_property_l1804_180467


namespace gumball_probability_l1804_180483

/-- Given a jar with pink and blue gumballs, if the probability of drawing two blue
    gumballs in a row with replacement is 16/36, then the probability of drawing
    a pink gumball is 1/3. -/
theorem gumball_probability (p_blue p_pink : ℝ) : 
  p_blue + p_pink = 1 →
  p_blue ^ 2 = 16 / 36 →
  p_pink = 1 / 3 :=
by sorry

end gumball_probability_l1804_180483


namespace product_ratio_integer_l1804_180464

def divisible_count (seq : List Nat) (d : Nat) : Nat :=
  (seq.filter (fun x => x % d == 0)).length

theorem product_ratio_integer (m n : List Nat) :
  (∀ d : Nat, d > 1 → divisible_count m d ≥ divisible_count n d) →
  m.all (· > 0) →
  n.all (· > 0) →
  n.length > 0 →
  ∃ k : Nat, k > 0 ∧ (m.prod : Int) = k * (n.prod : Int) := by
  sorry

end product_ratio_integer_l1804_180464


namespace johns_age_doubles_l1804_180456

/-- Represents John's current age -/
def current_age : ℕ := 18

/-- Represents the number of years ago when John's age was half of a future age -/
def years_ago : ℕ := 5

/-- Represents the number of years until John's age is twice his age from five years ago -/
def years_until_double : ℕ := 8

/-- Theorem stating that in 8 years, John's age will be twice his age from five years ago -/
theorem johns_age_doubles : 
  2 * (current_age - years_ago) = current_age + years_until_double := by
  sorry

end johns_age_doubles_l1804_180456


namespace union_equals_interval_l1804_180418

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ -1}
def B : Set ℝ := {y : ℝ | y ≥ 1}

-- Define the interval [-1, +∞)
def interval : Set ℝ := {x : ℝ | x ≥ -1}

-- Theorem stating that the union of A and B is equal to the interval [-1, +∞)
theorem union_equals_interval : A ∪ B = interval := by sorry

end union_equals_interval_l1804_180418


namespace boys_fraction_l1804_180471

/-- In a class with boys and girls, prove that the fraction of boys is 2/3 given the conditions. -/
theorem boys_fraction (tall_boys : ℕ) (total_boys : ℕ) (girls : ℕ) 
  (h1 : tall_boys = 18)
  (h2 : tall_boys = 3 * total_boys / 4)
  (h3 : girls = 12) :
  total_boys / (total_boys + girls) = 2 / 3 := by
  sorry

end boys_fraction_l1804_180471


namespace license_plate_palindrome_probability_l1804_180404

/-- Represents a license plate with 4 digits and 2 letters -/
structure LicensePlate where
  digits : Fin 10 → Fin 10
  letters : Fin 2 → Fin 26

/-- Checks if a sequence of 4 digits is a palindrome -/
def isPalindrome4 (s : Fin 4 → Fin 10) : Prop :=
  s 0 = s 3 ∧ s 1 = s 2

/-- Checks if a sequence of 2 letters is a palindrome -/
def isPalindrome2 (s : Fin 2 → Fin 26) : Prop :=
  s 0 = s 1

/-- The probability of a license plate containing at least one palindrome sequence -/
def palindromeProbability : ℚ :=
  5 / 104

/-- The main theorem stating the probability of a license plate containing at least one palindrome sequence -/
theorem license_plate_palindrome_probability :
  palindromeProbability = 5 / 104 := by
  sorry


end license_plate_palindrome_probability_l1804_180404


namespace factorial_20_divisibility_l1804_180422

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highest_power_dividing (base k : ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1) / base) 0

theorem factorial_20_divisibility : 
  (highest_power_dividing 12 6 20 = 6) ∧ 
  (highest_power_dividing 10 4 20 = 4) := by
  sorry

end factorial_20_divisibility_l1804_180422


namespace students_in_score_range_l1804_180496

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  total_students : ℕ
  mean : ℝ
  variance : ℝ
  prob_above_140 : ℝ

/-- Calculates the number of students within a given score range -/
def students_in_range (dist : ScoreDistribution) (lower upper : ℝ) : ℕ :=
  sorry

theorem students_in_score_range (dist : ScoreDistribution) 
  (h1 : dist.total_students = 50)
  (h2 : dist.mean = 120)
  (h3 : dist.prob_above_140 = 0.2) :
  students_in_range dist 100 140 = 30 :=
sorry

end students_in_score_range_l1804_180496


namespace boat_speed_l1804_180445

/-- Given a boat that travels 11 km/h downstream and 5 km/h upstream, 
    its speed in still water is 8 km/h. -/
theorem boat_speed (downstream upstream : ℝ) 
  (h1 : downstream = 11) 
  (h2 : upstream = 5) : 
  (downstream + upstream) / 2 = 8 := by
  sorry

end boat_speed_l1804_180445


namespace f_max_value_l1804_180497

noncomputable def f (x : ℝ) : ℝ := x^6 / (x^12 + 3*x^9 - 6*x^6 + 12*x^3 + 27)

theorem f_max_value :
  (∀ x : ℝ, f x ≤ 1 / (12 * Real.sqrt 3)) ∧
  (∃ x : ℝ, f x = 1 / (12 * Real.sqrt 3)) :=
by sorry

end f_max_value_l1804_180497


namespace sum_of_digits_B_is_seven_l1804_180432

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def A : ℕ := digit_sum (4444^4444)

def B : ℕ := digit_sum A

theorem sum_of_digits_B_is_seven :
  digit_sum B = 7 :=
sorry

end sum_of_digits_B_is_seven_l1804_180432


namespace variation_problem_l1804_180453

theorem variation_problem (c : ℝ) (R S T : ℝ → ℝ) (t : ℝ) :
  (∀ t, R t = c * (S t)^2 / (T t)^2) →
  R 0 = 2 ∧ S 0 = 1 ∧ T 0 = 2 →
  R t = 50 ∧ T t = 5 →
  S t = 12.5 := by
sorry

end variation_problem_l1804_180453


namespace double_percentage_increase_l1804_180403

theorem double_percentage_increase (x : ℝ) : 
  (1 + x / 100)^2 = 1 + 44 / 100 → x = 20 := by
sorry

end double_percentage_increase_l1804_180403


namespace union_equals_B_exists_union_equals_intersection_l1804_180477

-- Define the sets A, B, and C
def A : Set ℝ := {x | |x - 1| < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x - 6 < 0}
def C : Set ℝ := {x | x^2 - 2*x - 15 < 0}

-- Statement 1
theorem union_equals_B (a : ℝ) : 
  A ∪ B a = B a ↔ a ∈ Set.Icc (-5 : ℝ) (-1 : ℝ) := by sorry

-- Statement 2
theorem exists_union_equals_intersection :
  ∃ a ∈ Set.Icc (-19/5 : ℝ) (-1 : ℝ), A ∪ B a = B a ∩ C := by sorry

end union_equals_B_exists_union_equals_intersection_l1804_180477


namespace f_five_not_unique_l1804_180472

/-- A function satisfying the given functional equation for all real x and y -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f (3 * x + y) + 3 * x * y = f (4 * x - y) + 3 * x^2 + 2

/-- The theorem stating that f(5) cannot be uniquely determined -/
theorem f_five_not_unique : 
  ¬ ∃ (a : ℝ), ∀ (f : ℝ → ℝ), FunctionalEquation f → f 5 = a :=
sorry

end f_five_not_unique_l1804_180472


namespace saras_remaining_money_l1804_180475

/-- Calculates Sara's remaining money after her first paycheck and expenses --/
theorem saras_remaining_money :
  let week1_hours : ℕ := 40
  let week1_rate : ℚ := 11.5
  let week2_regular_hours : ℕ := 40
  let week2_overtime_hours : ℕ := 10
  let week2_rate : ℚ := 12
  let overtime_multiplier : ℚ := 1.5
  let sales : ℚ := 1000
  let commission_rate : ℚ := 0.05
  let tax_rate : ℚ := 0.15
  let insurance_cost : ℚ := 60
  let misc_fees : ℚ := 20
  let tire_cost : ℚ := 410

  let week1_earnings := week1_hours * week1_rate
  let week2_regular_earnings := week2_regular_hours * week2_rate
  let week2_overtime_earnings := week2_overtime_hours * (week2_rate * overtime_multiplier)
  let total_hourly_earnings := week1_earnings + week2_regular_earnings + week2_overtime_earnings
  let commission := sales * commission_rate
  let total_earnings := total_hourly_earnings + commission
  let taxes := total_earnings * tax_rate
  let total_deductions := taxes + insurance_cost + misc_fees
  let net_earnings := total_earnings - total_deductions
  let remaining_money := net_earnings - tire_cost

  remaining_money = 504.5 :=
by sorry

end saras_remaining_money_l1804_180475


namespace wheel_rotation_on_moving_car_l1804_180434

/-- A wheel is a circular object that can rotate. -/
structure Wheel :=
  (radius : ℝ)
  (center : ℝ × ℝ)

/-- A car is a vehicle with wheels. -/
structure Car :=
  (wheels : List Wheel)

/-- Motion types that an object can exhibit. -/
inductive MotionType
  | Rotation
  | Translation
  | Other

/-- A moving car is a car with a velocity. -/
structure MovingCar extends Car :=
  (velocity : ℝ × ℝ)

/-- The motion type exhibited by a wheel on a moving car. -/
def wheelMotionType (mc : MovingCar) (w : Wheel) : MotionType :=
  sorry

/-- Theorem: The wheels of a moving car exhibit rotational motion. -/
theorem wheel_rotation_on_moving_car (mc : MovingCar) (w : Wheel) 
  (h : w ∈ mc.wheels) : 
  wheelMotionType mc w = MotionType.Rotation :=
sorry

end wheel_rotation_on_moving_car_l1804_180434


namespace sports_cards_pages_l1804_180476

/-- Calculates the number of pages needed for a given number of cards and cards per page -/
def pagesNeeded (cards : ℕ) (cardsPerPage : ℕ) : ℕ :=
  (cards + cardsPerPage - 1) / cardsPerPage

theorem sports_cards_pages : 
  let baseballCards := 12
  let baseballCardsPerPage := 4
  let basketballCards := 14
  let basketballCardsPerPage := 3
  let soccerCards := 7
  let soccerCardsPerPage := 5
  (pagesNeeded baseballCards baseballCardsPerPage) +
  (pagesNeeded basketballCards basketballCardsPerPage) +
  (pagesNeeded soccerCards soccerCardsPerPage) = 10 := by
  sorry


end sports_cards_pages_l1804_180476


namespace construction_company_stone_order_l1804_180407

/-- The weight of stone ordered by a construction company -/
theorem construction_company_stone_order
  (concrete : ℝ) (bricks : ℝ) (total : ℝ)
  (h1 : concrete = 0.16666666666666666)
  (h2 : bricks = 0.16666666666666666)
  (h3 : total = 0.8333333333333334) :
  total - (concrete + bricks) = 0.5 := by
sorry

end construction_company_stone_order_l1804_180407


namespace lauren_mail_count_l1804_180417

/-- The number of pieces of mail Lauren sent on Monday -/
def monday_mail : ℕ := 65

/-- The number of pieces of mail Lauren sent on Tuesday -/
def tuesday_mail : ℕ := monday_mail + 10

/-- The number of pieces of mail Lauren sent on Wednesday -/
def wednesday_mail : ℕ := tuesday_mail - 5

/-- The number of pieces of mail Lauren sent on Thursday -/
def thursday_mail : ℕ := wednesday_mail + 15

/-- The total number of pieces of mail Lauren sent over the four days -/
def total_mail : ℕ := monday_mail + tuesday_mail + wednesday_mail + thursday_mail

theorem lauren_mail_count : total_mail = 295 := by
  sorry

end lauren_mail_count_l1804_180417


namespace equation_solution_l1804_180446

theorem equation_solution (x : ℝ) : 
  x ≠ 3 → ((2 - x) / (x - 3) = 1 / (x - 3) - 2) → x = 5 :=
by sorry

end equation_solution_l1804_180446


namespace inequality_proof_l1804_180499

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ((a^3) / (a^3 + 15*b*c*d))^(1/2) ≥ (a^(15/8)) / (a^(15/8) + b^(15/8) + c^(15/8) + d^(15/8)) := by
  sorry

end inequality_proof_l1804_180499


namespace smallest_positive_e_value_l1804_180444

theorem smallest_positive_e_value (a b c d e : ℤ) :
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -3 ∨ x = 4 ∨ x = 8 ∨ x = -1/4) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∃ a' b' c' d' : ℤ, ∀ x : ℚ, a' * x^4 + b' * x^3 + c' * x^2 + d' * x + e' = 0 ↔ 
      x = -3 ∨ x = 4 ∨ x = 8 ∨ x = -1/4) →
    e ≤ e') →
  e = 96 :=
by sorry

end smallest_positive_e_value_l1804_180444


namespace centroid_quadrilateral_area_ratio_l1804_180493

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is inside a quadrilateral -/
def isInterior (p : Point) (q : Quadrilateral) : Prop := sorry

/-- Calculates the centroid of a triangle -/
def centroid (a b c : Point) : Point := sorry

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Main theorem -/
theorem centroid_quadrilateral_area_ratio 
  (ABCD : Quadrilateral) 
  (P : Point) 
  (h1 : isConvex ABCD) 
  (h2 : isInterior P ABCD) : 
  let G1 := centroid ABCD.A ABCD.B P
  let G2 := centroid ABCD.B ABCD.C P
  let G3 := centroid ABCD.C ABCD.D P
  let G4 := centroid ABCD.D ABCD.A P
  let centroidQuad : Quadrilateral := ⟨G1, G2, G3, G4⟩
  area centroidQuad / area ABCD = 1 / 9 := by sorry

end centroid_quadrilateral_area_ratio_l1804_180493


namespace distance_traveled_proof_l1804_180490

/-- Calculates the distance traveled given initial and final odometer readings -/
def distance_traveled (initial_reading final_reading : Real) : Real :=
  final_reading - initial_reading

/-- Theorem stating that the distance traveled is 159.7 miles -/
theorem distance_traveled_proof (initial_reading final_reading : Real) 
  (h1 : initial_reading = 212.3)
  (h2 : final_reading = 372.0) :
  distance_traveled initial_reading final_reading = 159.7 := by
  sorry

end distance_traveled_proof_l1804_180490


namespace dereks_current_dogs_l1804_180431

/-- Represents the number of dogs and cars Derek has at different ages -/
structure DereksPets where
  dogs_at_six : ℕ
  cars_at_six : ℕ
  cars_bought : ℕ
  current_dogs : ℕ

/-- Theorem stating the conditions and the result to be proven -/
theorem dereks_current_dogs (d : DereksPets) 
  (h1 : d.dogs_at_six = 3 * d.cars_at_six)
  (h2 : d.dogs_at_six = 90)
  (h3 : d.cars_bought = 210)
  (h4 : d.cars_at_six + d.cars_bought = 2 * d.current_dogs) :
  d.current_dogs = 120 := by
  sorry

end dereks_current_dogs_l1804_180431


namespace power_series_expansion_of_exp_l1804_180419

open Real

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the n-th term of the power series
noncomputable def power_series_term (a : ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  (log a)^n / (Nat.factorial n : ℝ) * x^n

-- Theorem statement
theorem power_series_expansion_of_exp (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, f a x = ∑' n, power_series_term a n x :=
sorry

end power_series_expansion_of_exp_l1804_180419


namespace goods_train_speed_calculation_l1804_180491

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 36

/-- The speed of the express train in km/h -/
def express_train_speed : ℝ := 90

/-- The time difference between the departures of the two trains in hours -/
def time_difference : ℝ := 6

/-- The time taken by the express train to catch up with the goods train in hours -/
def catch_up_time : ℝ := 4

theorem goods_train_speed_calculation :
  goods_train_speed * (catch_up_time + time_difference) = express_train_speed * catch_up_time :=
sorry

end goods_train_speed_calculation_l1804_180491


namespace percentage_of_cat_owners_l1804_180401

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) : 
  total_students = 500 → cat_owners = 75 → 
  (cat_owners : ℚ) / (total_students : ℚ) * 100 = 15 := by
  sorry

end percentage_of_cat_owners_l1804_180401


namespace xyz_product_l1804_180433

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 360)
  (eq2 : y * (z + x) = 405)
  (eq3 : z * (x + y) = 450) :
  x * y * z = 2433 := by
sorry

end xyz_product_l1804_180433


namespace water_in_large_bottle_sport_formulation_l1804_180447

/-- Represents a flavored drink formulation -/
structure Formulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard_formulation : Formulation :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_formulation : Formulation :=
  { flavoring := 1, corn_syrup := 4, water := 60 }

/-- The amount of corn syrup in the large bottle (in ounces) -/
def large_bottle_corn_syrup : ℚ := 8

/-- Theorem stating the amount of water in the large bottle of sport formulation -/
theorem water_in_large_bottle_sport_formulation :
  (large_bottle_corn_syrup * sport_formulation.water) / sport_formulation.corn_syrup = 120 := by
  sorry

end water_in_large_bottle_sport_formulation_l1804_180447


namespace tub_drain_time_l1804_180492

/-- Given a tub that drains at a constant rate, this function calculates the additional time
    needed to empty the tub completely after a certain fraction has been drained. -/
def additional_drain_time (initial_fraction : ℚ) (initial_time : ℚ) : ℚ :=
  let remaining_fraction := 1 - initial_fraction
  let drain_rate := initial_fraction / initial_time
  remaining_fraction / drain_rate

/-- Theorem stating that for a tub draining 5/7 of its content in 4 minutes,
    it will take an additional 11.2 minutes to empty completely. -/
theorem tub_drain_time : additional_drain_time (5/7) 4 = 11.2 := by
  sorry

end tub_drain_time_l1804_180492


namespace bacteria_growth_l1804_180457

/-- The time interval between bacterial divisions in minutes -/
def division_interval : ℕ := 20

/-- The total observation time in hours -/
def total_time : ℕ := 3

/-- The number of divisions that occur in the total observation time -/
def num_divisions : ℕ := (total_time * 60) / division_interval

/-- The final number of bacteria after the total observation time -/
def final_bacteria_count : ℕ := 2^num_divisions

theorem bacteria_growth :
  final_bacteria_count = 512 :=
sorry

end bacteria_growth_l1804_180457


namespace bugs_eating_flowers_l1804_180413

theorem bugs_eating_flowers :
  let bug_amounts : List ℝ := [2.5, 3, 1.5, 2, 4, 0.5, 3]
  bug_amounts.sum = 16.5 := by
sorry

end bugs_eating_flowers_l1804_180413


namespace polygon_sides_proof_l1804_180478

theorem polygon_sides_proof (n : ℕ) : 
  let sides1 := n
  let sides2 := n + 4
  let sides3 := n + 12
  let sides4 := n + 13
  let diagonals (m : ℕ) := m * (m - 3) / 2
  diagonals sides1 + diagonals sides4 = diagonals sides2 + diagonals sides3 → n = 3 := by
  sorry

end polygon_sides_proof_l1804_180478


namespace divisibility_condition_l1804_180440

/-- A pair of positive integers (m, n) satisfies the divisibility condition if and only if
    it is of the form (k^2 + 1, k) or (k, k^2 + 1) for some positive integer k. -/
theorem divisibility_condition (m n : ℕ+) : 
  (∃ d : ℕ+, d * (m * n - 1) = (n^2 - n + 1)^2) ↔ 
  (∃ k : ℕ+, (m = k^2 + 1 ∧ n = k) ∨ (m = k ∧ n = k^2 + 1)) :=
sorry

end divisibility_condition_l1804_180440


namespace continuous_compound_interest_rate_l1804_180450

/-- Continuous compound interest rate calculation -/
theorem continuous_compound_interest_rate 
  (P : ℝ) -- Principal amount
  (A : ℝ) -- Total amount after interest
  (t : ℝ) -- Time in years
  (h1 : P = 600)
  (h2 : A = 760)
  (h3 : t = 4)
  : ∃ r : ℝ, (A = P * Real.exp (r * t)) ∧ (abs (r - 0.05909725) < 0.00000001) :=
sorry

end continuous_compound_interest_rate_l1804_180450


namespace geometric_sequence_property_l1804_180494

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) (k : ℝ) :
  isGeometricSequence a →
  a 5 * a 8 * a 11 = k →
  k^2 = a 5 * a 6 * a 7 * a 9 * a 10 * a 11 :=
by
  sorry

end geometric_sequence_property_l1804_180494


namespace car_sale_profit_l1804_180423

def original_price : ℕ := 50000
def loss_percentage : ℚ := 10 / 100
def gain_percentage : ℚ := 20 / 100

def friend_selling_price : ℕ := 54000

theorem car_sale_profit (original_price : ℕ) (loss_percentage gain_percentage : ℚ) 
  (friend_selling_price : ℕ) : 
  let man_selling_price : ℚ := (1 - loss_percentage) * original_price
  let friend_buying_price : ℚ := man_selling_price
  (1 + gain_percentage) * friend_buying_price = friend_selling_price := by
  sorry

end car_sale_profit_l1804_180423


namespace congruence_theorem_l1804_180484

theorem congruence_theorem (x : ℤ) 
  (h1 : (8 + x) % 8 = 27 % 8)
  (h2 : (10 + x) % 27 = 16 % 27)
  (h3 : (13 + x) % 125 = 36 % 125) :
  x % 120 = 11 := by
sorry

end congruence_theorem_l1804_180484


namespace marble_distribution_l1804_180439

theorem marble_distribution (capacity_second : ℝ) : 
  capacity_second > 0 →
  capacity_second + (3/4 * capacity_second) = 1050 →
  capacity_second = 600 := by
sorry

end marble_distribution_l1804_180439


namespace eulers_formula_l1804_180412

theorem eulers_formula (x : ℝ) : Complex.exp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x := by
  sorry

end eulers_formula_l1804_180412


namespace new_person_weight_l1804_180479

/-- Proves that the weight of a new person is 65 kg given the conditions of the problem -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 45 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 65 :=
by sorry

end new_person_weight_l1804_180479


namespace parabola_focus_line_intersection_l1804_180458

/-- Represents a parabola with equation y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line passing through the focus of a parabola -/
structure FocusLine where
  angle : ℝ
  h_angle_eq : angle = π / 4

/-- Represents the intersection points of a line with a parabola -/
structure Intersection where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The main theorem -/
theorem parabola_focus_line_intersection
  (para : Parabola) (line : FocusLine) (inter : Intersection) :
  let midpoint_x := (inter.A.1 + inter.B.1) / 2
  let axis_distance := midpoint_x - para.p / 2
  axis_distance = 4 → para.p = 2 := by
  sorry

end parabola_focus_line_intersection_l1804_180458


namespace equal_probability_for_first_ace_l1804_180466

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ace_count : ℕ)
  (hc : ace_count ≤ total_cards)

/-- Represents a card game -/
structure CardGame :=
  (deck : Deck)
  (player_count : ℕ)
  (hpc : player_count > 0)

/-- The probability of a player receiving the first Ace -/
def first_ace_probability (game : CardGame) (player : ℕ) : ℚ :=
  1 / game.player_count

/-- Theorem stating that in a specific card game, each player has an equal probability of receiving the first Ace -/
theorem equal_probability_for_first_ace (game : CardGame) 
    (h1 : game.deck.total_cards = 32) 
    (h2 : game.deck.ace_count = 4) 
    (h3 : game.player_count = 4) : 
    ∀ (player : ℕ), player > 0 → player ≤ game.player_count → 
    first_ace_probability game player = 1 / 8 := by
  sorry

#check equal_probability_for_first_ace

end equal_probability_for_first_ace_l1804_180466


namespace discount_percentage_calculation_l1804_180495

theorem discount_percentage_calculation (original_price : ℝ) 
  (john_tip_rate : ℝ) (jane_tip_rate : ℝ) (price_difference : ℝ) 
  (h1 : original_price = 24.00000000000002)
  (h2 : john_tip_rate = 0.15)
  (h3 : jane_tip_rate = 0.15)
  (h4 : price_difference = 0.36)
  (h5 : original_price * (1 + john_tip_rate) - 
        original_price * (1 - D) * (1 + jane_tip_rate) = price_difference) :
  D = price_difference / (original_price * (1 + john_tip_rate)) := by
sorry

#eval (0.36 / 27.600000000000024 : Float)

end discount_percentage_calculation_l1804_180495


namespace max_charge_at_150_l1804_180461

-- Define the charge function
def charge (x : ℝ) : ℝ := 1000 * x - 5 * (x - 100)^2

-- State the theorem
theorem max_charge_at_150 :
  ∀ x ∈ Set.Icc 100 180,
    charge x ≤ charge 150 ∧
    charge 150 = 112500 := by
  sorry

-- Note: Set.Icc 100 180 represents the closed interval [100, 180]

end max_charge_at_150_l1804_180461


namespace train_length_l1804_180436

/-- Given a train that can cross an electric pole in 120 seconds while traveling at 90 km/h,
    prove that its length is 3000 meters. -/
theorem train_length (crossing_time : ℝ) (speed_kmh : ℝ) (length : ℝ) : 
  crossing_time = 120 →
  speed_kmh = 90 →
  length = speed_kmh * (1000 / 3600) * crossing_time →
  length = 3000 := by
  sorry

end train_length_l1804_180436


namespace removed_triangles_area_l1804_180481

/-- Given a square with side length s, from which isosceles right triangles
    with equal sides of length x are removed from each corner to form a rectangle
    with longer side 16 units, the total area of the four removed triangles is 512 square units. -/
theorem removed_triangles_area (s x : ℝ) : 
  s > 0 ∧ x > 0 ∧ s - x = 16 ∧ 2 * x^2 = (s - 2*x)^2 → 4 * (1/2 * x^2) = 512 := by
  sorry

#check removed_triangles_area

end removed_triangles_area_l1804_180481


namespace workshop_employees_l1804_180441

theorem workshop_employees :
  ∃ (n k1 k2 : ℕ),
    0 < n ∧ n < 60 ∧
    n = 8 * k1 + 5 ∧
    n = 6 * k2 + 3 ∧
    (n = 21 ∨ n = 45) :=
by sorry

end workshop_employees_l1804_180441


namespace max_reflections_is_nine_l1804_180425

/-- Represents the angle between two mirrors in degrees -/
def mirror_angle : ℝ := 10

/-- Represents the increase in angle of incidence after each reflection in degrees -/
def angle_increase : ℝ := 10

/-- Represents the maximum angle at which reflection is possible in degrees -/
def max_reflection_angle : ℝ := 90

/-- Calculates the angle of incidence after a given number of reflections -/
def angle_after_reflections (n : ℕ) : ℝ := n * angle_increase

/-- Determines if reflection is possible after a given number of reflections -/
def is_reflection_possible (n : ℕ) : Prop :=
  angle_after_reflections n ≤ max_reflection_angle

/-- The maximum number of reflections possible -/
def max_reflections : ℕ := 9

/-- Theorem stating that the maximum number of reflections is 9 -/
theorem max_reflections_is_nine :
  (∀ n : ℕ, is_reflection_possible n → n ≤ max_reflections) ∧
  is_reflection_possible max_reflections ∧
  ¬is_reflection_possible (max_reflections + 1) :=
sorry

end max_reflections_is_nine_l1804_180425


namespace cubic_real_root_l1804_180482

/-- Given a cubic polynomial with real coefficients c and d, 
    if -3 - 4i is a root, then the real root is 25/3 -/
theorem cubic_real_root (c d : ℝ) : 
  (c * (Complex.I ^ 3 + (-3 - 4*Complex.I) ^ 3) + 
   4 * (Complex.I ^ 2 + (-3 - 4*Complex.I) ^ 2) + 
   d * (Complex.I + (-3 - 4*Complex.I)) - 100 = 0) →
  (∃ (x : ℝ), c * x^3 + 4 * x^2 + d * x - 100 = 0 ∧ x = 25/3) :=
by sorry

end cubic_real_root_l1804_180482


namespace point_coordinates_l1804_180437

/-- Given point M(5, -6) and vector a = (1, -2), if NM = 3a, then N has coordinates (2, 0) -/
theorem point_coordinates (M N : ℝ × ℝ) (a : ℝ × ℝ) : 
  M = (5, -6) → 
  a = (1, -2) → 
  N.1 - M.1 = 3 * a.1 ∧ N.2 - M.2 = 3 * a.2 → 
  N = (2, 0) := by
  sorry

end point_coordinates_l1804_180437


namespace secretary_project_hours_l1804_180451

theorem secretary_project_hours (total_hours : ℕ) (ratio_1 ratio_2 ratio_3 ratio_4 : ℕ) :
  total_hours = 2080 →
  ratio_1 = 3 →
  ratio_2 = 5 →
  ratio_3 = 7 →
  ratio_4 = 11 →
  (ratio_1 + ratio_2 + ratio_3 + ratio_4) * (total_hours / (ratio_1 + ratio_2 + ratio_3 + ratio_4)) = total_hours →
  ratio_4 * (total_hours / (ratio_1 + ratio_2 + ratio_3 + ratio_4)) = 880 :=
by sorry

end secretary_project_hours_l1804_180451


namespace canoe_kayak_ratio_l1804_180462

/-- Represents the rental prices and quantities of canoes and kayaks --/
structure RentalInfo where
  canoePrice : ℕ
  kayakPrice : ℕ
  canoeCount : ℕ
  kayakCount : ℕ

/-- Calculates the total revenue from canoe and kayak rentals --/
def totalRevenue (info : RentalInfo) : ℕ :=
  info.canoePrice * info.canoeCount + info.kayakPrice * info.kayakCount

/-- Theorem stating the ratio of canoes to kayaks given the rental conditions --/
theorem canoe_kayak_ratio (info : RentalInfo) :
  info.canoePrice = 15 →
  info.kayakPrice = 18 →
  totalRevenue info = 405 →
  info.canoeCount = info.kayakCount + 5 →
  (info.canoeCount : ℚ) / info.kayakCount = 3 / 2 := by
  sorry


end canoe_kayak_ratio_l1804_180462
